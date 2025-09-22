#!/usr/bin/env python3
"""
pynautobot Script: Find Devices with Interfaces Connected to Circuits via Cables

This script follows Nautobot best practices to:
1. Find all devices in the environment
2. Get interfaces for those devices
3. Filter for interfaces that are connected to circuits through cables
4. Generate a CSV file with device, interface_name, and circuit_name

The script uses pynautobot's filtering capabilities and follows the established patterns
for querying related objects efficiently.

THIS WAS CREATED ENTIRELY USING DOCUMENTATION FROM NETWORK TO CODE REPOS AS KB.
"""

import csv
import os
import sys
from typing import Dict, List

import pynautobot  # type: ignore


class NautobotCircuitInterfaceExtractor:
    """
    A class to extract device interfaces connected to circuits via cables.

    This follows Nautobot best practices for API interactions and data processing.
    """

    def __init__(self, url: str, token: str):
        """
        Initialize the Nautobot API connection.

        Args:
            url: Nautobot API URL (e.g., "https://nautobot.example.com")
            token: API token for authentication
        """
        self.nautobot = pynautobot.api(url, token=token)
        print(f"Connected to Nautobot at {url}")

    def extract_circuit_connected_interfaces(self) -> List[Dict[str, str]]:
        """
        Extract device interfaces connected to circuits via cables.

        This method uses a comprehensive approach to find all interfaces
        that are connected to circuit terminations through cables.

        Returns:
            List of dictionaries with device, interface_name, and circuit_name
        """
        results = []

        print("Starting extraction of interfaces connected to circuits...")

        # Step 1: Get all interfaces that have cables attached
        print("Fetching interfaces with cables...")
        interfaces_with_cables = self.nautobot.dcim.interfaces.filter(
            cable__isnull=False
        )
        print(f"Found {len(interfaces_with_cables)} interfaces with cables")

        if not interfaces_with_cables:
            print("No interfaces with cables found")
            return results

        # Step 2: Get all circuit terminations
        print("Fetching circuit terminations...")
        circuit_terminations = self.nautobot.circuits.circuit_terminations.all()
        print(f"Found {len(circuit_terminations)} circuit terminations")

        if not circuit_terminations:
            print("No circuit terminations found")
            return results

        # Create a mapping of circuit termination IDs to circuits
        ct_to_circuit = {}
        for ct in circuit_terminations:
            ct_to_circuit[ct.id] = ct.circuit

        # Step 3: Check each interface to see if it connects to a circuit
        print("Checking interface connections to circuits...")

        for interface in interfaces_with_cables:
            try:
                # Get the cable for this interface
                cable = interface.cable
                if not cable:
                    continue

                # Check what this interface is connected to
                # The interface could be either termination_a or termination_b
                connected_endpoint = None

                # Determine which termination is the interface and get the other endpoint
                if hasattr(cable, "termination_a") and hasattr(cable, "termination_b"):
                    # Check if this interface is termination_a
                    if (
                        hasattr(cable.termination_a, "id")
                        and cable.termination_a.id == interface.id
                    ):
                        connected_endpoint = cable.termination_b
                    # Check if this interface is termination_b
                    elif (
                        hasattr(cable.termination_b, "id")
                        and cable.termination_b.id == interface.id
                    ):
                        connected_endpoint = cable.termination_a

                # Check if the connected endpoint is a circuit termination
                if connected_endpoint and hasattr(connected_endpoint, "id"):
                    if connected_endpoint.id in ct_to_circuit:
                        # This interface is connected to a circuit!
                        circuit = ct_to_circuit[connected_endpoint.id]

                        results.append(
                            {
                                "device": interface.device.name,
                                "interface_name": interface.name,
                                "circuit_name": circuit.cid
                                if hasattr(circuit, "cid")
                                else str(circuit),
                            }
                        )

                        print(
                            f"Found connection: {interface.device.name} -> {interface.name} -> {circuit.cid}"
                        )

            except Exception as e:
                print(f"Error processing interface {interface.name}: {e}")
                continue

        print(f"Found {len(results)} interfaces connected to circuits")
        return results

    def alternative_cable_based_approach(self) -> List[Dict[str, str]]:
        """
        Alternative approach: Start with cables and check their endpoints.

        Returns:
            List of dictionaries with device, interface_name, and circuit_name
        """
        results = []

        print("Alternative approach: Starting with all cables...")

        # Get all cables
        cables = self.nautobot.dcim.cables.all()
        print(f"Found {len(cables)} total cables")

        for cable in cables:
            try:
                termination_a = cable.termination_a
                termination_b = cable.termination_b

                interface_endpoint = None
                circuit_endpoint = None

                # Check if termination_a is an interface and termination_b is a circuit termination
                if (
                    hasattr(termination_a, "device")
                    and hasattr(termination_a, "name")
                    and hasattr(termination_b, "circuit")
                ):
                    interface_endpoint = termination_a
                    circuit_endpoint = termination_b

                # Check if termination_b is an interface and termination_a is a circuit termination
                elif (
                    hasattr(termination_b, "device")
                    and hasattr(termination_b, "name")
                    and hasattr(termination_a, "circuit")
                ):
                    interface_endpoint = termination_b
                    circuit_endpoint = termination_a

                # If we found an interface-to-circuit connection
                if interface_endpoint and circuit_endpoint:
                    circuit = circuit_endpoint.circuit

                    results.append(
                        {
                            "device": interface_endpoint.device.name,
                            "interface_name": interface_endpoint.name,
                            "circuit_name": circuit.cid
                            if hasattr(circuit, "cid")
                            else str(circuit),
                        }
                    )

            except Exception as e:
                print(f"Error processing cable {cable.id}: {e}")
                continue

        print(
            f"Alternative approach found {len(results)} interfaces connected to circuits"
        )
        return results

    def save_to_csv(
        self,
        data: List[Dict[str, str]],
        filename: str = "devices_with_circuit_interfaces.csv",
    ):
        """
        Save the results to a CSV file.

        Args:
            data: List of dictionaries with device, interface_name, circuit_name
            filename: Output CSV filename
        """
        if not data:
            print("No data to save")
            return

        print(f"Saving {len(data)} records to {filename}")

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["device", "interface_name", "circuit_name"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write data rows
            for row in data:
                writer.writerow(row)

        print(f"Data saved to {filename}")


def main():
    """
    Main function to execute the script.
    """
    # Configuration - Update these values for your environment
    NAUTOBOT_URL = os.getenv("NAUTOBOT_URL", "https://your-nautobot-instance.com")
    NAUTOBOT_TOKEN = os.getenv("NAUTOBOT_TOKEN", "your-api-token-here")

    # Validate configuration
    if not NAUTOBOT_URL or NAUTOBOT_URL == "https://your-nautobot-instance.com":
        print(
            "Error: Please set NAUTOBOT_URL environment variable or update the script"
        )
        print("Example: export NAUTOBOT_URL='https://your-nautobot-instance.com'")
        sys.exit(1)

    if not NAUTOBOT_TOKEN or NAUTOBOT_TOKEN == "your-api-token-here":
        print(
            "Error: Please set NAUTOBOT_TOKEN environment variable or update the script"
        )
        print("Example: export NAUTOBOT_TOKEN='your-api-token-here'")
        sys.exit(1)

    try:
        # Initialize the extractor
        extractor = NautobotCircuitInterfaceExtractor(NAUTOBOT_URL, NAUTOBOT_TOKEN)

        # Try the primary approach first
        print("\n=== Primary Approach: Interface-based search ===")
        results_primary = extractor.extract_circuit_connected_interfaces()

        # Try the alternative approach
        print("\n=== Alternative Approach: Cable-based search ===")
        results_alternative = extractor.alternative_cable_based_approach()

        # Use the approach that found more results, or combine them
        if len(results_alternative) > len(results_primary):
            print(
                f"\nUsing alternative approach (found {len(results_alternative)} vs {len(results_primary)} results)"
            )
            final_results = results_alternative
        else:
            print(
                f"\nUsing primary approach (found {len(results_primary)} vs {len(results_alternative)} results)"
            )
            final_results = results_primary

        # Remove duplicates based on device + interface + circuit combination
        unique_results = []
        seen = set()
        for result in final_results:
            key = (result["device"], result["interface_name"], result["circuit_name"])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        print(
            f"\nFinal results: {len(unique_results)} unique device-interface-circuit connections"
        )

        # Display results
        if unique_results:
            print("\nResults summary:")
            print("=" * 60)
            print(f"{'Device':<20} {'Interface':<15} {'Circuit':<20}")
            print("=" * 60)

            for i, result in enumerate(unique_results[:10]):  # Show first 10 results
                print(
                    f"{result['device']:<20} {result['interface_name']:<15} {result['circuit_name']:<20}"
                )

            if len(unique_results) > 10:
                print(f"... and {len(unique_results) - 10} more")

            print("=" * 60)
        else:
            print("\nNo interfaces connected to circuits were found.")
            print("This could mean:")
            print("1. No devices have interfaces connected to circuits via cables")
            print("2. The circuit terminations are not properly configured")
            print(
                "3. The cables are not properly connected between interfaces and circuit terminations"
            )

        # Save to CSV regardless (even if empty, for consistency)
        extractor.save_to_csv(unique_results)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
