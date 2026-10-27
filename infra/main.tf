# Databricks next to the Brussels lakehouse (github.com/Huntsama/brussels-lakehouse).
# The lake and its resource group belong to that repo; here they are only read, so this state never touches ingest.
terraform {
  required_providers {
    azurerm    = { source = "hashicorp/azurerm", version = "~> 4.0" }
    databricks = { source = "databricks/databricks", version = "~> 1.50" }
  }
}

provider "azurerm" {
  features {}
  # Registering every provider fails on student subscriptions.
  resource_provider_registrations = "none"
  resource_providers_to_register  = ["Microsoft.Databricks"]
  subscription_id                 = var.subscription_id
}

# Signs in with `az login`, like azurerm.
provider "databricks" {
  host                        = azurerm_databricks_workspace.dbw.workspace_url
  azure_workspace_resource_id = azurerm_databricks_workspace.dbw.id
}

variable "subscription_id" { type = string }
variable "name" { default = "belakehouse" }
# The repo is private: a fine-grained GitHub token with Contents read/write on this repo only, so the Git folder can
# clone and you can commit from the workspace. Pass it as TF_VAR_github_token, never in terraform.tfvars.
variable "github_token" {
  type      = string
  sensitive = true
}

data "azurerm_resource_group" "rg" { name = "rg-${var.name}" }

data "azurerm_storage_account" "lake" {
  name                = "${var.name}lake"
  resource_group_name = data.azurerm_resource_group.rg.name
}

locals {
  location = data.azurerm_resource_group.rg.location
  abfss    = { for c in ["bronze", "silver", "gold"] : c => "abfss://${c}@${data.azurerm_storage_account.lake.name}.dfs.core.windows.net/" }
}

resource "azurerm_databricks_workspace" "dbw" {
  name                        = "dbw-${var.name}"
  resource_group_name         = data.azurerm_resource_group.rg.name
  location                    = local.location
  sku                         = "trial" # 14 days of free DBUs; then "premium" (Unity Catalog needs it), changes in place
  managed_resource_group_name = "rg-${var.name}-dbw-managed"
  # Clusters get public IPs instead of a NAT gateway, which bills ~30 EUR a month even with nothing running.
  custom_parameters { no_public_ip = false }
}

# The identity Unity Catalog uses to read bronze and write Delta tables back to the lake.
resource "azurerm_databricks_access_connector" "lake" {
  name                = "dbac-${var.name}"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = local.location
  identity { type = "SystemAssigned" }
}

resource "azurerm_role_assignment" "connector_writes_lake" {
  scope                = data.azurerm_storage_account.lake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_databricks_access_connector.lake.identity[0].principal_id
}

# Unity Catalog: one credential, one external location per lake container, and a catalog whose tables live in gold.
resource "databricks_storage_credential" "lake" {
  name = "lake"
  azure_managed_identity { access_connector_id = azurerm_databricks_access_connector.lake.id }
  depends_on = [azurerm_role_assignment.connector_writes_lake]
}

resource "databricks_external_location" "layer" {
  for_each        = local.abfss
  name            = each.key
  url             = each.value
  credential_name = databricks_storage_credential.lake.name
}

resource "databricks_catalog" "brussels" {
  name         = "brussels"
  storage_root = "${local.abfss["gold"]}uc"
  depends_on   = [databricks_external_location.layer]
}

resource "databricks_schema" "layer" {
  for_each     = toset(["bronze", "silver", "gold", "ml"])
  catalog_name = databricks_catalog.brussels.name
  name         = each.key
}

# One small cluster with the ML runtime (scikit-learn, LightGBM, MLflow preinstalled).
# Standard_DS3_v2 = 4 cores, the whole DSv2 quota of a student subscription.
data "databricks_current_user" "me" {}

data "databricks_spark_version" "ml" {
  ml                = true
  long_term_support = true
}

locals {
  single_node_conf = {
    "spark.databricks.cluster.profile" = "singleNode"
    "spark.master"                     = "local[*]"
  }
}

# For the notebooks and lessons. Stops itself after 15 idle minutes.
resource "databricks_cluster" "single" {
  cluster_name            = "single-node"
  spark_version           = data.databricks_spark_version.ml.id
  node_type_id            = "Standard_DS3_v2"
  autotermination_minutes = 15
  num_workers             = 0
  data_security_mode      = "SINGLE_USER"
  single_user_name        = data.databricks_current_user.me.user_name
  spark_conf              = local.single_node_conf
  custom_tags             = { ResourceClass = "SingleNode" }
}

# This repo as a Git folder in the workspace: write the lessons there, commit back to GitHub.
resource "databricks_git_credential" "github" {
  git_provider          = "gitHub"
  git_username          = "Huntsama"
  personal_access_token = var.github_token
}

resource "databricks_repo" "this" {
  url        = "https://github.com/Huntsama/brussels-transit-ml.git"
  path       = "${data.databricks_current_user.me.repos}/brussels-transit-ml"
  depends_on = [databricks_git_credential.github]
}

# The pipeline notebooks the job runs, uploaded from notebooks/ on every apply that changes them.
resource "databricks_notebook" "pipeline" {
  for_each = fileset("${path.module}/../notebooks", "*.{sql,py}")
  source   = "${path.module}/../notebooks/${each.key}"
  path     = "/Shared/brussels/${trimsuffix(trimsuffix(each.key, ".sql"), ".py")}"
}

# Silver then gold every morning, on a cluster that exists only while the job runs.
# 05:30 so it never overlaps the notebook cluster: two 4-core clusters don't fit the 6-core quota.
resource "databricks_job" "lakehouse" {
  name = "lakehouse-silver-gold"

  schedule {
    quartz_cron_expression = "0 30 5 * * ?"
    timezone_id            = "Europe/Brussels"
  }

  job_cluster {
    job_cluster_key = "single"
    new_cluster {
      spark_version      = data.databricks_spark_version.ml.id
      node_type_id       = "Standard_DS3_v2"
      num_workers        = 0
      data_security_mode = "SINGLE_USER"
      single_user_name   = data.databricks_current_user.me.user_name
      spark_conf         = local.single_node_conf
      custom_tags        = { ResourceClass = "SingleNode" }
    }
  }

  task {
    task_key        = "silver"
    job_cluster_key = "single"
    notebook_task { notebook_path = databricks_notebook.pipeline["01_silver.sql"].path }
  }

  task {
    task_key        = "gold"
    job_cluster_key = "single"
    depends_on { task_key = "silver" }
    notebook_task { notebook_path = databricks_notebook.pipeline["02_gold.sql"].path }
  }

  email_notifications { on_failure = [data.databricks_current_user.me.user_name] }
}

# Live status every 15 minutes: process what Kafka buffered since the last run, then stop (no always-on cluster).
# Paused by default: unpause in the Databricks UI while the producer runs (infra/streaming.tf).
resource "databricks_job" "live_status" {
  name = "stream-live-status"
  schedule {
    quartz_cron_expression = "0 0/15 6-23 * * ?"
    timezone_id            = "Europe/Brussels"
    pause_status           = "PAUSED"
  }
  job_cluster {
    job_cluster_key = "single"
    new_cluster {
      spark_version      = data.databricks_spark_version.ml.id
      node_type_id       = "Standard_DS3_v2"
      num_workers        = 0
      data_security_mode = "SINGLE_USER"
      single_user_name   = data.databricks_current_user.me.user_name
      spark_conf         = local.single_node_conf
      custom_tags        = { ResourceClass = "SingleNode" }
    }
  }
  task {
    task_key        = "live"
    job_cluster_key = "single"
    notebook_task {
      notebook_path   = databricks_notebook.pipeline["03_stream_live_status.py"].path
      base_parameters = { trigger = "available_now" }
    }
  }
}

output "databricks_url" { value = "https://${azurerm_databricks_workspace.dbw.workspace_url}" }
output "cluster_id" { value = databricks_cluster.single.id }
output "job_url" { value = databricks_job.lakehouse.url }

# MLOps: retrain the grid-load model every Monday on fresh Elia + weather data, on Databricks (not a laptop).
# The notebook logs the run to MLflow and registers brussels.ml.grid_load_lgbm, promoted to @champion only if better.
resource "databricks_job" "train_grid_load" {
  name = "train-grid-load"
  schedule {
    quartz_cron_expression = "0 0 6 ? * MON"
    timezone_id            = "Europe/Brussels"
    pause_status           = "PAUSED" # unpause in the UI after the first manual run works
  }
  job_cluster {
    job_cluster_key = "single"
    new_cluster {
      spark_version      = data.databricks_spark_version.ml.id
      node_type_id       = "Standard_DS3_v2"
      num_workers        = 0
      data_security_mode = "SINGLE_USER"
      single_user_name   = data.databricks_current_user.me.user_name
      spark_conf         = local.single_node_conf
      custom_tags        = { ResourceClass = "SingleNode" }
    }
  }
  task {
    task_key        = "data"
    job_cluster_key = "single"
    spark_python_task { python_file = "${databricks_repo.this.path}/tasks/grid_load/get_data.py" }
  }
  task {
    task_key        = "train"
    job_cluster_key = "single"
    depends_on { task_key = "data" }
    spark_python_task { python_file = "${databricks_repo.this.path}/tasks/grid_load/notebook.py" }
    library {
      pypi { package = "holidays" } # the ML runtime already has lightgbm, torch and mlflow
    }
  }
  email_notifications { on_failure = [data.databricks_current_user.me.user_name] }
}

output "train_grid_load_job" { value = databricks_job.train_grid_load.url }
