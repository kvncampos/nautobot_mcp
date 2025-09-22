# Nautobot Circuit Interface Extractor

This directory contains example scripts for working with Nautobot using pynautobot.

## demo_pynautobot_sample.py

A comprehensive script that finds all devices with interfaces connected to circuits via cables and generates a CSV report.

### Purpose

This script addresses a common network documentation requirement: identifying which device interfaces are connected to WAN circuits through physical cable connections. This is useful for:

- Network documentation and inventory
- Circuit impact analysis
- Cable management
- Compliance reporting

### Features

- **Comprehensive Search**: Uses two complementary approaches to ensure all connections are found
- **Best Practices**: Follows pynautobot patterns and Nautobot API best practices
- **Error Handling**: Robust error handling with detailed logging
- **CSV Output**: Generates a clean CSV file with results
- **Duplicate Removal**: Automatically removes duplicate entries

### Requirements

- Python 3.6+
- pynautobot library
- Access to a Nautobot instance with API token

### Installation

```bash
pip install pynautobot
```

### Configuration

Set environment variables:

```bash
export NAUTOBOT_URL="https://your-nautobot-instance.com"
export NAUTOBOT_TOKEN="your-api-token-here"
```

Or edit the script directly to set the URL and token values.

### Usage

```bash
python demo_pynautobot_sample.py
```

### Output

The script generates a CSV file named `devices_with_circuit_interfaces.csv` with the following structure:

```csv
device,interface_name,circuit_name
device1,GigabitEthernet0/1,CIRCUIT-001
device2,TenGigabitEthernet0/0/0,CIRCUIT-002
```

### How It Works

The script uses two complementary approaches:

#### Primary Approach: Interface-Based Search
1. Finds all interfaces that have cables attached
2. Gets all circuit terminations
3. For each interface, checks if its cable connects to a circuit termination
4. Records the device, interface, and circuit information

#### Alternative Approach: Cable-Based Search
1. Examines all cables in the system
2. Identifies cables that connect interfaces to circuit terminations
3. Extracts the relevant information

The script automatically selects the approach that finds more results and removes duplicates.

### Data Model Understanding

In Nautobot:
- **Devices** have **Interfaces**
- **Circuits** have **Circuit Terminations**
- **Cables** connect different endpoints, including:
  - Interface ↔ Circuit Termination
  - Interface ↔ Interface
  - Other component types

This script specifically looks for `Interface ↔ Circuit Termination` connections.

### Troubleshooting

#### No Results Found

If the script returns no results, check:

1. **Cables are properly connected**: In Nautobot, cables must be explicitly created between interfaces and circuit terminations
2. **Circuit terminations exist**: Circuits must have terminations configured
3. **API permissions**: Ensure your API token has read access to devices, interfaces, cables, and circuits
4. **Data model**: Verify that your Nautobot instance has the expected data structure

#### Common Issues

1. **Connection Error**: Check NAUTOBOT_URL and network connectivity
2. **Authentication Error**: Verify NAUTOBOT_TOKEN is correct and has necessary permissions
3. **Import Error**: Install pynautobot with `pip install pynautobot`

### Best Practices Demonstrated

This script demonstrates several Nautobot/pynautobot best practices:

1. **Efficient API Usage**: Uses filtering to reduce API calls
2. **Error Handling**: Comprehensive try/catch blocks with meaningful error messages
3. **Resource Management**: Proper handling of API responses
4. **Documentation**: Clear code documentation and comments
5. **Modular Design**: Organized into logical methods and classes
6. **Configuration Management**: Environment variable support

### Extending the Script

You can extend this script to:

- Add more fields to the CSV output (device location, circuit provider, etc.)
- Filter by device type, location, or circuit attributes
- Generate different output formats (JSON, Excel, etc.)
- Add email notifications or integration with other systems
- Include cable information (cable type, length, etc.)

### Example Extensions

#### Add Device Location to Output

```python
results.append({
    'device': interface.device.name,
    'device_location': interface.device.location.name if interface.device.location else 'N/A',
    'interface_name': interface.name,
    'circuit_name': circuit.cid,
    'circuit_provider': circuit.provider.name if circuit.provider else 'N/A'
})
```

#### Filter by Device Role

```python
# Only include devices with specific roles
target_roles = ['router', 'switch']
interfaces_with_cables = self.nautobot.dcim.interfaces.filter(
    cable__isnull=False,
    device__role__in=target_roles
)
```

### Related Documentation

- [Nautobot API Documentation](https://nautobot.readthedocs.io/en/stable/user-guide/platform-functionality/rest-api/)
- [pynautobot Documentation](https://pynautobot.readthedocs.io/)
- [Nautobot Cable Model](https://nautobot.readthedocs.io/en/stable/user-guide/core-data-model/dcim/cable/)
- [Nautobot Circuit Model](https://nautobot.readthedocs.io/en/stable/user-guide/core-data-model/circuits/)
