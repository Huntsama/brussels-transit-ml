# Streaming: Azure Event Hubs with its Kafka endpoint (Standard tier; Basic has no Kafka), and the producer
# running as one small container on Azure Container Apps.
# Cost while it exists: Event Hubs Standard ~0.03 EUR/hour (~22 EUR/month) + the container ~5 EUR/month.
# Run it for the demo weeks, then: terraform destroy -target=azurerm_eventhub_namespace.stream -target=azurerm_container_app.producer

resource "azurerm_eventhub_namespace" "stream" {
  name                = "evh-${var.name}"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  sku                 = "Standard"
  capacity            = 1
}

resource "azurerm_eventhub" "topic" {
  for_each          = toset(["stib-waiting", "stib-vehicles"])
  name              = each.key
  namespace_id      = azurerm_eventhub_namespace.stream.id
  partition_count   = 2
  message_retention = 1 # days: Databricks writes everything to Delta, Kafka only buffers
}

resource "azurerm_eventhub_namespace_authorization_rule" "send" {
  name                = "producer"
  namespace_name      = azurerm_eventhub_namespace.stream.name
  resource_group_name = data.azurerm_resource_group.rg.name
  send                = true
}

resource "azurerm_eventhub_namespace_authorization_rule" "listen" {
  name                = "databricks"
  namespace_name      = azurerm_eventhub_namespace.stream.name
  resource_group_name = data.azurerm_resource_group.rg.name
  listen              = true
}

# Databricks reads the listen key from a secret scope, never from the notebook
resource "databricks_secret_scope" "stream" {
  name = "stream"
}

resource "databricks_secret" "listen" {
  scope        = databricks_secret_scope.stream.name
  key          = "eventhubs-connection"
  string_value = azurerm_eventhub_namespace_authorization_rule.listen.primary_connection_string
}

resource "azurerm_container_app_environment" "env" {
  name                = "cae-${var.name}"
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
}

resource "azurerm_container_app" "producer" {
  name                         = "stib-producer"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = data.azurerm_resource_group.rg.name
  revision_mode                = "Single"

  secret {
    name  = "kafka-connection"
    value = azurerm_eventhub_namespace_authorization_rule.send.primary_connection_string
  }

  template {
    min_replicas = 1
    max_replicas = 1
    container {
      name   = "producer"
      image  = "ghcr.io/huntsama/stib-producer:latest" # built and pushed by .github/workflows/ci.yml
      cpu    = 0.25
      memory = "0.5Gi"
      env {
        name  = "KAFKA_BOOTSTRAP"
        value = "${azurerm_eventhub_namespace.stream.name}.servicebus.windows.net:9093"
      }
      env {
        name        = "KAFKA_CONNECTION"
        secret_name = "kafka-connection"
      }
    }
  }
}

output "eventhubs_bootstrap" { value = "${azurerm_eventhub_namespace.stream.name}.servicebus.windows.net:9093" }
output "eventhubs_send_connection" {
  value     = azurerm_eventhub_namespace_authorization_rule.send.primary_connection_string
  sensitive = true
}

# The notebook publishes line_status.json into the static-website container, next to the dashboard
resource "databricks_external_location" "web" {
  name            = "web"
  url             = "abfss://$web@${data.azurerm_storage_account.lake.name}.dfs.core.windows.net/"
  credential_name = databricks_storage_credential.lake.name
}

# The API: scales to zero when nobody calls it, so it costs nothing while idle
resource "azurerm_container_app" "api" {
  name                         = "transit-api"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = data.azurerm_resource_group.rg.name
  revision_mode                = "Single"

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0
    max_replicas = 2
    container {
      name   = "api"
      image  = "ghcr.io/huntsama/transit-api:latest"
      cpu    = 0.25
      memory = "0.5Gi"
      env {
        name  = "DATA_BASE_URL"
        value = "https://${data.azurerm_storage_account.lake.name}.z1.web.core.windows.net"
      }
    }
  }
}

output "api_url" { value = "https://${azurerm_container_app.api.ingress[0].fqdn}/docs" }
