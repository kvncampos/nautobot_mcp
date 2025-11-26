# Examples

This page provides real-world examples of using the Nautobot MCP Server with GitHub Copilot.

## Basic Examples

### Example 1: Discovering API Endpoints

**Scenario:** You want to find endpoints related to IP address management.

**Query to Copilot:**
```
What API endpoints are available for managing IP addresses in Nautobot?
```

**What Happens:**

1. Copilot uses `nautobot_openapi_api_request_schema` tool
2. Semantic search finds relevant endpoints
3. Returns ranked list with descriptions

**Expected Response:**
```
I found several endpoints for IP address management:

1. GET /api/ipam/ip-addresses/
   List all IP addresses

2. POST /api/ipam/ip-addresses/
   Create a new IP address

3. GET /api/ipam/ip-addresses/{id}/
   Retrieve a specific IP address

4. PATCH /api/ipam/ip-addresses/{id}/
   Update an IP address

5. DELETE /api/ipam/ip-addresses/{id}/
   Delete an IP address
```

---

### Example 2: Querying Nautobot Data

**Scenario:** You need to see all active devices in a specific location.

**Query to Copilot:**
```
Get all active devices in the Main DC location
```

**What Happens:**

1. Copilot uses `nautobot_dynamic_api_request` tool
2. Constructs GET request with filters
3. Executes against your Nautobot instance
4. Formats response

**Expected Response:**
```
Found 15 active devices in Main DC:

1. core-switch-01 (Cisco Catalyst 9300)
2. core-switch-02 (Cisco Catalyst 9300)
3. access-switch-01 (Cisco Catalyst 2960)
...
```

---

### Example 3: Knowledge Base Search

**Scenario:** You want to learn about Nautobot jobs.

**Query to Copilot:**
```
How do I create a custom Nautobot job?
```

**What Happens:**

1. Copilot uses `nautobot_kb_semantic_search` tool
2. Searches indexed documentation and code
3. Returns relevant sections with sources

**Expected Response:**
```
Based on the Nautobot documentation, here's how to create a custom job:

1. Create a job class inheriting from `Job`:

```python
from nautobot.extras.jobs import Job

class MyCustomJob(Job):
    class Meta:
        name = "My Custom Job"
        description = "Description of what this job does"
    
    def run(self, *args, **kwargs):
        # Your job logic here
        self.logger.info("Job is running")
```

2. Place the file in your app's `jobs.py` or jobs directory
3. Register it in Nautobot

Source: nautobot/nautobot/docs/user-guides/jobs.md
```

---

## Intermediate Examples

### Example 4: Creating Resources

**Scenario:** Create a new device in Nautobot.

**Query to Copilot:**
```
Create a new device named "test-switch-01" with device type "cisco-catalyst-3850", 
in the Main DC location, with status active
```

**What Happens:**

1. Copilot finds the appropriate endpoint
2. Constructs POST request with required fields
3. Executes the API call
4. Returns the created resource

**Expected Interaction:**

```
I'll create that device for you. First, let me find the correct endpoint and required fields...

Using POST /api/dcim/devices/ with the following data:
{
  "name": "test-switch-01",
  "device_type": "cisco-catalyst-3850",
  "location": "main-dc",
  "status": "active"
}

✅ Device created successfully!
ID: 12345
Name: test-switch-01
Status: active
Location: Main DC
```

---

### Example 5: Updating Resources

**Scenario:** Change a device's status to maintenance mode.

**Query to Copilot:**
```
Change the status of device "core-switch-01" to maintenance
```

**What Happens:**

1. Query device to get ID
2. Use PATCH to update status
3. Confirm the change

**Expected Response:**
```
I'll update the device status for you.

Step 1: Finding device ID...
Found: core-switch-01 (ID: 789)

Step 2: Updating status to maintenance...
PATCH /api/dcim/devices/789/
{ "status": "maintenance" }

✅ Status updated successfully!
Device: core-switch-01
Previous Status: active
New Status: maintenance
```

---

### Example 6: Complex Queries with Filters

**Scenario:** Find all Cisco switches in active status across all locations.

**Query to Copilot:**
```
Show me all Cisco switches that are currently active, grouped by location
```

**What Happens:**

1. Query devices with manufacturer and status filters
2. Group results by location
3. Format output

**Expected Response:**
```
Found 24 active Cisco switches across 3 locations:

Main DC (15 switches):
- core-switch-01 (Catalyst 9300)
- core-switch-02 (Catalyst 9300)
- access-switch-01 (Catalyst 2960)
...

Branch Office 1 (6 switches):
- branch1-switch-01 (Catalyst 2960)
...

Branch Office 2 (3 switches):
- branch2-switch-01 (Catalyst 2960)
...
```

---

## Advanced Examples

### Example 7: Multi-Step Workflow

**Scenario:** Onboard a new device with IP assignment.

**Query to Copilot:**
```
I need to onboard a new device:
1. Create device "new-switch-01" as Cisco Catalyst 3850 in Main DC
2. Create an interface "GigabitEthernet1/0/1" on this device
3. Assign IP 192.168.1.10/24 to the interface
```

**What Happens:**

Copilot orchestrates multiple API calls:

**Expected Interaction:**
```
I'll help you onboard the new device step by step.

Step 1: Creating device...
POST /api/dcim/devices/
✅ Device created (ID: 1001)

Step 2: Creating interface...
POST /api/dcim/interfaces/
{
  "device": 1001,
  "name": "GigabitEthernet1/0/1",
  "type": "1000base-t"
}
✅ Interface created (ID: 5001)

Step 3: Assigning IP address...
POST /api/ipam/ip-addresses/
{
  "address": "192.168.1.10/24",
  "assigned_object_type": "dcim.interface",
  "assigned_object_id": 5001
}
✅ IP address assigned

Summary:
- Device: new-switch-01 (ID: 1001)
- Interface: GigabitEthernet1/0/1 (ID: 5001)
- IP: 192.168.1.10/24

Device onboarding complete!
```

---

### Example 8: Data Validation

**Scenario:** Validate device configuration before deployment.

**Query to Copilot:**
```
Check if device "core-switch-01" has all required properties:
- Device role must be "core-switch"
- Must have at least 2 power connections
- Must have a primary IP assigned
```

**What Happens:**

1. Fetch device details
2. Validate each requirement
3. Report findings

**Expected Response:**
```
Checking device "core-switch-01"...

✅ Device Role: core-switch (Valid)
❌ Power Connections: 1 found, 2 required
✅ Primary IP: 10.0.0.1 (Assigned)

Validation Result: 2/3 checks passed

Recommendation: Add a second power connection to meet requirements.
```

---

### Example 9: Documentation Research

**Scenario:** Research before implementing a feature.

**Query to Copilot:**
```
I need to implement GraphQL queries in my application. 
Show me:
1. Documentation about the GraphQL API
2. Authentication requirements
3. Example queries from the knowledge base
```

**What Happens:**

1. Search knowledge base for GraphQL docs
2. Find authentication patterns
3. Locate example code

**Expected Response:**
```
Here's what I found about Nautobot's GraphQL API:

## Documentation:
The GraphQL API is available at /api/graphql/ and provides a flexible 
way to query Nautobot data...
(Source: nautobot/nautobot/docs/user-guides/graphql.md)

## Authentication:
GraphQL uses the same token authentication as the REST API:
```
Authorization: Token your-api-token-here
```
(Source: nautobot/nautobot/docs/user-guides/graphql.md)

## Example Query:
```graphql
query {
  devices(name: "core-switch-01") {
    id
    name
    device_type {
      model
      manufacturer {
        name
      }
    }
    location {
      name
    }
  }
}
```
(Source: nautobot/nautobot/examples/graphql_examples.py)

## Python Example:
```python
import requests

url = "http://nautobot/api/graphql/"
headers = {"Authorization": "Token your-token"}
query = """
{
  devices { 
    name 
    status 
  }
}
"""

response = requests.post(url, json={"query": query}, headers=headers)
```
(Source: nautobot/nautobot/examples/api_examples.py)
```

---

### Example 10: Repository Management

**Scenario:** Add custom plugin to knowledge base.

**Query to Copilot:**
```
Add the repository "myorg/nautobot-custom-plugin" to the knowledge base 
so I can search its documentation
```

**What Happens:**

1. Copilot uses `nautobot_kb_add_repo` tool
2. Repository is cloned
3. Content is indexed
4. Confirms completion

**Expected Response:**
```
I'll add that repository to the knowledge base.

Adding: myorg/nautobot-custom-plugin
Cloning repository... ✓
Indexing files... ✓
Creating embeddings... ✓

Repository added successfully!
- Documents indexed: 45
- File types: .py, .md, .rst
- Time taken: 32s

You can now search this repository's content using knowledge base queries.
```

---

## Integration Examples

### Example 11: VS Code Workflow

**Scenario:** Using MCP while developing code.

**In VS Code:**

1. **Write code with inline suggestions:**
   ```python
   # Comment: Get all devices from Nautobot
   # Copilot suggests:
   import pynautobot
   nb = pynautobot.api(url="http://nautobot", token="token")
   devices = nb.dcim.devices.all()
   ```

2. **Ask in Chat:**
   ```
   What's the correct way to filter devices by location in pynautobot?
   ```

3. **Get contextual help:**
   ```
   Copilot searches knowledge base and returns:
   
   devices = nb.dcim.devices.filter(location="main-dc")
   # Or using location ID:
   devices = nb.dcim.devices.filter(location_id="uuid-here")
   
   (Source: pynautobot documentation)
   ```

---

### Example 12: Testing API Endpoints

**Scenario:** Test an endpoint before using it in code.

**Query:**
```
Test the device list endpoint with a limit of 3 and 
show me the response structure
```

**Response:**
```
Testing GET /api/dcim/devices/?limit=3

Response (200 OK):
```json
{
  "count": 150,
  "next": "http://nautobot/api/dcim/devices/?limit=3&offset=3",
  "previous": null,
  "results": [
    {
      "id": "uuid1",
      "name": "core-switch-01",
      "device_type": {...},
      "location": {...},
      "status": "active",
      "primary_ip": {...}
    },
    // ... 2 more devices
  ]
}
```

The response structure shows:
- Pagination info (count, next, previous)
- Results array with device objects
- Each device has id, name, device_type, location, status, primary_ip
```

---

## Real-World Scenarios

### Scenario 1: Network Audit

**Task:** Audit all devices for compliance.

**Queries:**

```
1. "Get all devices without a primary IP"
2. "Find devices with status 'planned' older than 90 days"
3. "List devices missing required custom fields"
```

### Scenario 2: IP Address Management

**Task:** Manage IP address allocation.

**Queries:**

```
1. "Show available IP addresses in the 10.0.0.0/24 prefix"
2. "Find all unassigned IP addresses"
3. "Create IP address 10.0.0.50/24 with DNS name server01.example.com"
```

### Scenario 3: Plugin Development

**Task:** Develop a custom Nautobot plugin.

**Queries:**

```
1. "Show me the plugin architecture documentation"
2. "What are the best practices for plugin development?"
3. "Find examples of plugins that add custom views"
4. "How do I add a custom model to my plugin?"
```

---

## Tips for Effective Use

### Be Specific

❌ "Tell me about devices"
✅ "Show me all Cisco devices in active status with their IP addresses"

### Provide Context

❌ "Create a device"
✅ "Create a device named 'switch-01', type 'cisco-3850', in 'Main DC', status 'active'"

### Chain Operations

❌ Multiple separate queries
✅ "First find the location ID for 'Main DC', then create a device in that location"

### Verify Before Destructive Operations

Always review before:
- DELETE operations
- Bulk updates
- Production changes

---

## Next Steps

- [Tools Reference](tools.md) - Detailed tool documentation
- [Troubleshooting](troubleshooting.md) - Common issues
- [Architecture](architecture.md) - How it works
