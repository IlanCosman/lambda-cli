use anyhow::{anyhow, Context, Result};
use reqwest::header::AUTHORIZATION;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;

pub const API_BASE_URL: &str = "https://cloud.lambdalabs.com/api/v1";
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// The machine image family used when launching an instance without an explicit
/// image. Lambda's own default (sending no image) is the latest Lambda Stack on
/// Ubuntu 22.04; we pin to the Ubuntu 24.04 Lambda Stack instead.
pub const DEFAULT_IMAGE_FAMILY: &str = "lambda-stack-24-04";

/// Resolve an optional image argument to the family that will actually be used.
///
/// Blank or whitespace-only input is treated as "not specified" so it falls back
/// to [`DEFAULT_IMAGE_FAMILY`] (rather than sending an invalid empty family);
/// surrounding whitespace on a real value is trimmed.
pub fn resolve_image_family(image: Option<&str>) -> &str {
    image
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_IMAGE_FAMILY)
}

#[derive(Error, Debug)]
pub enum LambdaError {
    #[error("API key not set. Set LAMBDA_API_KEY or LAMBDA_API_KEY_COMMAND environment variable")]
    ApiKeyNotSet,
    #[error("Failed to execute API key command: {0}")]
    ApiKeyCommandFailed(String),
    #[error("Instance type '{0}' not found")]
    InstanceTypeNotFound(String),
    #[error("No regions available for instance type '{0}'")]
    NoRegionsAvailable(String),
    #[error("No instance IDs returned from launch request")]
    NoInstanceIds,
    #[error("API request failed: {0}")]
    ApiError(String),
    #[error("SSH key is required for this operation")]
    SshKeyRequired,
}

#[derive(Deserialize, Debug)]
pub struct ApiResponse<T> {
    pub data: T,
}

#[derive(Deserialize, Debug)]
pub struct ApiErrorResponse {
    pub error: ApiErrorDetail,
}

#[derive(Deserialize, Debug)]
pub struct ApiErrorDetail {
    pub message: String,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct Instance {
    pub id: Option<String>,
    pub name: Option<String>,
    pub status: Option<String>,
    pub ip: Option<String>,
    pub ssh_key_names: Option<Vec<String>>,
    pub instance_type: Option<InstanceTypeInfo>,
    pub region: Option<RegionInfo>,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct InstanceTypeInfo {
    pub name: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct RegionInfo {
    pub name: Option<String>,
}

/// Filesystem (persistent storage) information
#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct Filesystem {
    pub id: String,
    pub name: String,
    pub mount_point: String,
    pub created: String,
    pub region: FilesystemRegion,
    pub is_in_use: bool,
    #[serde(default)]
    pub bytes_used: u64,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct FilesystemRegion {
    pub name: String,
    pub description: String,
}

#[derive(Deserialize, Debug)]
pub struct LaunchResponse {
    pub instance_ids: Vec<String>,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct InstanceTypeData {
    pub name: String,
    pub description: String,
    pub price_cents_per_hour: i32,
    pub vcpus: u32,
    pub memory_gib: u32,
    pub storage_gib: u32,
    pub regions_available: Vec<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct InstanceTypeResponse {
    pub instance_type: InstanceType,
    pub regions_with_capacity_available: Vec<Region>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct InstanceType {
    pub description: String,
    pub price_cents_per_hour: i32,
    pub specs: InstanceSpecs,
}

#[derive(Deserialize, Debug, Clone)]
pub struct InstanceSpecs {
    pub vcpus: u32,
    pub memory_gib: u32,
    pub storage_gib: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Region {
    pub name: String,
    #[allow(dead_code)]
    pub description: String,
}

/// A machine image available in Lambda Cloud (returned by the images endpoint)
#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct Image {
    pub id: String,
    pub name: String,
    pub description: String,
    pub family: String,
    pub version: String,
    pub architecture: String,
    pub region: ImageRegion,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct ImageRegion {
    pub name: String,
    pub description: String,
}

/// A summary of a single image family, aggregated across regions and versions.
/// The `family` is the value users pass to select an image at launch.
#[derive(Debug, Clone, Serialize)]
pub struct ImageFamilySummary {
    pub family: String,
    pub description: String,
    pub architectures: Vec<String>,
    pub regions: Vec<String>,
}

/// Collapse the raw image list into one entry per family, with the set of
/// architectures and regions each family is available in. Sorted by family name.
pub fn summarize_image_families(images: &[Image]) -> Vec<ImageFamilySummary> {
    use std::collections::{BTreeMap, BTreeSet};

    struct Acc {
        description: String,
        architectures: BTreeSet<String>,
        regions: BTreeSet<String>,
    }

    let mut families: BTreeMap<String, Acc> = BTreeMap::new();
    for image in images {
        let acc = families.entry(image.family.clone()).or_insert_with(|| Acc {
            description: image.description.clone(),
            architectures: BTreeSet::new(),
            regions: BTreeSet::new(),
        });
        // Prefer the shortest description within a family (avoids version-suffixed variants).
        if image.description.len() < acc.description.len() {
            acc.description = image.description.clone();
        }
        acc.architectures.insert(image.architecture.clone());
        acc.regions.insert(image.region.name.clone());
    }

    families
        .into_iter()
        .map(|(family, acc)| ImageFamilySummary {
            family,
            description: acc.description,
            architectures: acc.architectures.into_iter().collect(),
            regions: acc.regions.into_iter().collect(),
        })
        .collect()
}

/// Source for the API key - either a direct value or a command to execute
#[derive(Debug, Clone)]
enum ApiKeySource {
    /// Direct API key value (already resolved)
    Direct(String),
    /// Command to execute to get the API key (lazy evaluation)
    Command(String),
}

/// Lambda API client
pub struct LambdaClient {
    client: Client,
    api_key_source: ApiKeySource,
    /// Cached API key (used for lazy evaluation)
    cached_api_key: Mutex<Option<String>>,
}

impl LambdaClient {
    pub fn new(api_key: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            api_key_source: ApiKeySource::Direct(api_key),
            cached_api_key: Mutex::new(None),
        })
    }

    /// Create a client with a lazy API key source (command executed on first use)
    fn new_lazy(api_key_source: ApiKeySource) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            api_key_source,
            cached_api_key: Mutex::new(None),
        })
    }

    /// Create a client using environment variables for the API key.
    ///
    /// Checks in order:
    /// 1. `LAMBDA_API_KEY` - Direct API key
    /// 2. `LAMBDA_API_KEY_COMMAND` - Command to execute to get the API key (e.g., `op read op://vault/lambda/api-key`)
    ///
    /// By default, if `LAMBDA_API_KEY_COMMAND` is used, the command is executed immediately.
    pub fn from_env() -> Result<Self> {
        Self::from_env_with_options(false)
    }

    /// Create a client using environment variables for the API key with options.
    ///
    /// If `lazy` is true and `LAMBDA_API_KEY_COMMAND` is used, the command execution
    /// is deferred until the first API request.
    pub fn from_env_with_options(lazy: bool) -> Result<Self> {
        // First, try direct API key (always immediate)
        if let Ok(key) = std::env::var("LAMBDA_API_KEY") {
            if !key.is_empty() {
                return Self::new(key);
            }
        }

        // Then, try command-based retrieval
        if let Ok(command) = std::env::var("LAMBDA_API_KEY_COMMAND") {
            if !command.is_empty() {
                if lazy {
                    // Defer command execution until first API request
                    return Self::new_lazy(ApiKeySource::Command(command));
                } else {
                    // Execute command immediately (default behavior)
                    let key = execute_api_key_command(&command)?;
                    return Self::new(key);
                }
            }
        }

        Err(LambdaError::ApiKeyNotSet.into())
    }

    /// Get the API key, executing the command if necessary (lazy evaluation)
    fn get_api_key(&self) -> Result<String> {
        match &self.api_key_source {
            ApiKeySource::Direct(key) => Ok(key.clone()),
            ApiKeySource::Command(cmd) => {
                let mut cache = self
                    .cached_api_key
                    .lock()
                    .map_err(|e| anyhow!("Failed to acquire lock: {}", e))?;

                if let Some(key) = cache.as_ref() {
                    return Ok(key.clone());
                }

                let key = execute_api_key_command(cmd)?;
                *cache = Some(key.clone());
                Ok(key)
            }
        }
    }

    /// Validate the API key by making a test request
    pub async fn validate_api_key(&self) -> Result<()> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/instances", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to connect to Lambda API")?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let error_msg = Self::parse_error_response(response).await;
            Err(anyhow!(
                "API key validation failed ({}): {}",
                status,
                error_msg
            ))
        }
    }

    /// List all available instance types
    pub async fn list_instance_types(&self) -> Result<Vec<InstanceTypeData>> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/instance-types", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch instance types")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to list instances: {}", error_msg));
        }

        let response: ApiResponse<HashMap<String, InstanceTypeResponse>> = response
            .json()
            .await
            .context("Failed to parse instance types response")?;

        let mut result: Vec<InstanceTypeData> = response
            .data
            .into_iter()
            .map(|(name, data)| InstanceTypeData {
                name,
                description: data.instance_type.description,
                price_cents_per_hour: data.instance_type.price_cents_per_hour,
                vcpus: data.instance_type.specs.vcpus,
                memory_gib: data.instance_type.specs.memory_gib,
                storage_gib: data.instance_type.specs.storage_gib,
                regions_available: data
                    .regions_with_capacity_available
                    .into_iter()
                    .map(|r| r.name)
                    .collect(),
            })
            .collect();

        result.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }

    /// List all available machine images
    pub async fn list_images(&self) -> Result<Vec<Image>> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/images", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch images")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to list images: {}", error_msg));
        }

        let response: ApiResponse<Vec<Image>> = response
            .json()
            .await
            .context("Failed to parse images response")?;

        Ok(response.data)
    }

    /// Get instance type details (for checking availability)
    pub async fn get_instance_type(&self, gpu: &str) -> Result<Option<InstanceTypeResponse>> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/instance-types", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch instance types")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to get instance types: {}", error_msg));
        }

        let response: ApiResponse<HashMap<String, InstanceTypeResponse>> = response
            .json()
            .await
            .context("Failed to parse instance types")?;

        Ok(response.data.get(gpu).cloned())
    }

    /// Launch a new instance
    pub async fn launch_instance(
        &self,
        gpu: &str,
        ssh_key: &str,
        name: Option<&str>,
        region: Option<&str>,
    ) -> Result<LaunchResult> {
        self.launch_instance_with_filesystem(gpu, ssh_key, name, region, None, None)
            .await
    }

    /// Launch a new instance with optional filesystem attachment and image family.
    ///
    /// When `image` is `None`, the instance is launched with
    /// [`DEFAULT_IMAGE_FAMILY`]. Pass an image family (e.g. `ubuntu-24-04`) to
    /// override it.
    pub async fn launch_instance_with_filesystem(
        &self,
        gpu: &str,
        ssh_key: &str,
        name: Option<&str>,
        region: Option<&str>,
        filesystem: Option<&str>,
        image: Option<&str>,
    ) -> Result<LaunchResult> {
        let instance_type_response = self
            .get_instance_type(gpu)
            .await?
            .ok_or_else(|| LambdaError::InstanceTypeNotFound(gpu.to_string()))?;

        let region_name = if let Some(r) = region {
            // Validate the specified region is available
            if !instance_type_response
                .regions_with_capacity_available
                .iter()
                .any(|reg| reg.name == r)
            {
                return Err(anyhow!(
                    "Region '{}' is not available for instance type '{}'. Available regions: {}",
                    r,
                    gpu,
                    instance_type_response
                        .regions_with_capacity_available
                        .iter()
                        .map(|reg| reg.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            r.to_string()
        } else {
            // Auto-select first available region
            instance_type_response
                .regions_with_capacity_available
                .first()
                .ok_or_else(|| LambdaError::NoRegionsAvailable(gpu.to_string()))?
                .name
                .clone()
        };

        let url = format!("{}/instance-operations/launch", API_BASE_URL);

        let image_family = resolve_image_family(image);
        let mut payload = serde_json::json!({
            "region_name": region_name,
            "instance_type_name": gpu,
            "ssh_key_names": [ssh_key],
            "quantity": 1,
            "image": { "family": image_family }
        });

        if let Some(instance_name) = name {
            payload["name"] = serde_json::Value::String(instance_name.to_string());
        }

        if let Some(fs_name) = filesystem {
            payload["file_system_names"] = serde_json::json!([fs_name]);
        }

        let api_key = self.get_api_key()?;
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .context("Failed to send launch request")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to launch instance: {}", error_msg));
        }

        let parsed_response: ApiResponse<LaunchResponse> = response
            .json()
            .await
            .context("Failed to parse launch response")?;

        let instance_id = parsed_response
            .data
            .instance_ids
            .first()
            .ok_or(LambdaError::NoInstanceIds)?
            .clone();

        Ok(LaunchResult {
            instance_id,
            region: region_name,
        })
    }

    /// Terminate an instance
    pub async fn terminate_instance(&self, instance_id: &str) -> Result<()> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/instance-operations/terminate", API_BASE_URL);
        let payload = serde_json::json!({
            "instance_ids": [instance_id]
        });

        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .context("Failed to send terminate request")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to terminate instance: {}", error_msg));
        }

        Ok(())
    }

    /// List all running instances
    pub async fn list_running_instances(&self) -> Result<Vec<Instance>> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/instances", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch running instances")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to list running instances: {}", error_msg));
        }

        let response: ApiResponse<Vec<Instance>> = response
            .json()
            .await
            .context("Failed to parse running instances response")?;

        Ok(response.data)
    }

    /// Get details for a specific instance
    pub async fn get_instance(&self, instance_id: &str) -> Result<Instance> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/instances/{}", API_BASE_URL, instance_id);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch instance details")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!(
                "Failed to get instance details ({}): {}",
                status,
                error_msg
            ));
        }

        let response: ApiResponse<Instance> = response
            .json()
            .await
            .context("Failed to parse instance details")?;

        Ok(response.data)
    }

    /// Check if a GPU type is available
    pub async fn check_availability(&self, gpu: &str) -> Result<Vec<String>> {
        let instance_type = self
            .get_instance_type(gpu)
            .await?
            .ok_or_else(|| LambdaError::InstanceTypeNotFound(gpu.to_string()))?;

        Ok(instance_type
            .regions_with_capacity_available
            .into_iter()
            .map(|r| r.name)
            .collect())
    }

    /// List all filesystems
    pub async fn list_filesystems(&self) -> Result<Vec<Filesystem>> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/file-systems", API_BASE_URL);
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to fetch filesystems")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to list filesystems: {}", error_msg));
        }

        let response: ApiResponse<Vec<Filesystem>> = response
            .json()
            .await
            .context("Failed to parse filesystems response")?;

        Ok(response.data)
    }

    /// Create a new filesystem
    pub async fn create_filesystem(&self, name: &str, region: &str) -> Result<Filesystem> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/file-systems", API_BASE_URL);
        let payload = serde_json::json!({
            "name": name,
            "region_name": region
        });

        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .context("Failed to create filesystem")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to create filesystem: {}", error_msg));
        }

        let response: ApiResponse<Filesystem> = response
            .json()
            .await
            .context("Failed to parse create filesystem response")?;

        Ok(response.data)
    }

    /// Delete a filesystem
    pub async fn delete_filesystem(&self, filesystem_id: &str) -> Result<()> {
        let api_key = self.get_api_key()?;
        let url = format!("{}/file-systems/{}", API_BASE_URL, filesystem_id);

        let response = self
            .client
            .delete(&url)
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .send()
            .await
            .context("Failed to delete filesystem")?;

        if !response.status().is_success() {
            let error_msg = Self::parse_error_response(response).await;
            return Err(anyhow!("Failed to delete filesystem: {}", error_msg));
        }

        Ok(())
    }

    async fn parse_error_response(response: reqwest::Response) -> String {
        response
            .json::<ApiErrorResponse>()
            .await
            .map(|e| e.error.message)
            .unwrap_or_else(|_| "Unknown error".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct LaunchResult {
    pub instance_id: String,
    pub region: String,
}

/// Execute a shell command to retrieve the API key.
fn execute_api_key_command(command: &str) -> Result<String> {
    use std::process::Command;

    let output = if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", command]).output()
    } else {
        Command::new("sh").args(["-c", command]).output()
    };

    match output {
        Ok(output) => {
            if output.status.success() {
                let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if key.is_empty() {
                    Err(LambdaError::ApiKeyCommandFailed(
                        "Command returned empty output".to_string(),
                    )
                    .into())
                } else {
                    Ok(key)
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(
                    LambdaError::ApiKeyCommandFailed(format!("Command failed: {}", stderr.trim()))
                        .into(),
                )
            }
        }
        Err(e) => Err(LambdaError::ApiKeyCommandFailed(format!(
            "Failed to execute command: {}",
            e
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_error_messages() {
        assert_eq!(
            LambdaError::ApiKeyNotSet.to_string(),
            "API key not set. Set LAMBDA_API_KEY or LAMBDA_API_KEY_COMMAND environment variable"
        );
        assert_eq!(
            LambdaError::InstanceTypeNotFound("gpu_1x_a100".to_string()).to_string(),
            "Instance type 'gpu_1x_a100' not found"
        );
    }

    #[test]
    fn test_api_base_url() {
        assert_eq!(API_BASE_URL, "https://cloud.lambdalabs.com/api/v1");
    }

    #[test]
    fn test_default_image_family() {
        assert_eq!(DEFAULT_IMAGE_FAMILY, "lambda-stack-24-04");
    }

    #[test]
    fn test_resolve_image_family() {
        // Unspecified or blank input falls back to the default.
        assert_eq!(resolve_image_family(None), DEFAULT_IMAGE_FAMILY);
        assert_eq!(resolve_image_family(Some("")), DEFAULT_IMAGE_FAMILY);
        assert_eq!(resolve_image_family(Some("   ")), DEFAULT_IMAGE_FAMILY);
        // A real value is used as-is, with surrounding whitespace trimmed.
        assert_eq!(resolve_image_family(Some("ubuntu-24-04")), "ubuntu-24-04");
        assert_eq!(
            resolve_image_family(Some("  ubuntu-24-04 ")),
            "ubuntu-24-04"
        );
    }

    fn test_image(family: &str, description: &str, architecture: &str, region: &str) -> Image {
        Image {
            id: "id".to_string(),
            name: "name".to_string(),
            description: description.to_string(),
            family: family.to_string(),
            version: "1.0".to_string(),
            architecture: architecture.to_string(),
            region: ImageRegion {
                name: region.to_string(),
                description: "desc".to_string(),
            },
        }
    }

    #[test]
    fn test_summarize_image_families() {
        let images = vec![
            test_image("ubuntu-24-04", "Ubuntu 24.04 LTS", "x86_64", "us-east-1"),
            test_image("ubuntu-24-04", "Ubuntu 24.04 LTS", "x86_64", "us-west-1"),
            test_image("ubuntu-24-04", "Ubuntu 24.04 LTS", "arm64", "us-east-3"),
            test_image("lambda-stack-24-04", "Lambda Stack", "x86_64", "us-east-1"),
        ];

        let summaries = summarize_image_families(&images);

        // Sorted by family name.
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].family, "lambda-stack-24-04");
        assert_eq!(summaries[1].family, "ubuntu-24-04");

        // Architectures and regions are deduplicated and sorted.
        assert_eq!(summaries[1].architectures, vec!["arm64", "x86_64"]);
        assert_eq!(
            summaries[1].regions,
            vec!["us-east-1", "us-east-3", "us-west-1"]
        );
    }

    #[test]
    fn test_summarize_prefers_shortest_description() {
        let images = vec![
            test_image(
                "lambda-stack-22-04",
                "AI ready GPU image 22.04.5",
                "x86_64",
                "us-east-1",
            ),
            test_image(
                "lambda-stack-22-04",
                "AI ready GPU image",
                "x86_64",
                "us-west-1",
            ),
        ];

        let summaries = summarize_image_families(&images);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].description, "AI ready GPU image");
    }
}
