#!/bin/bash
#
# Network Emulation Setup for Table 3
# Requires: sudo privileges, tc (traffic control)
#
# Usage:
#   sudo ./setup_network_emulation.sh lan    # Configure LAN (10 Gbps, 0.1 ms RTT)
#   sudo ./setup_network_emulation.sh wan    # Configure WAN (100 Mbps, 50 ms RTT)
#   sudo ./setup_network_emulation.sh reset  # Reset to default

set -e

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root (use sudo)"
   exit 1
fi

# Network interface (loopback for testing)
IFACE="lo"

# Function to reset network settings
reset_network() {
    echo "Resetting network emulation on $IFACE..."
    tc qdisc del dev $IFACE root 2>/dev/null || true
    echo "✓ Network emulation reset"
}

# Function to setup LAN emulation
setup_lan() {
    echo "Setting up LAN emulation on $IFACE..."
    echo "  - Bandwidth: 10 Gbps"
    echo "  - RTT: 0.1 ms (latency: 0.05 ms)"
    echo "  - No packet loss"

    # Reset first
    reset_network

    # Add qdisc with rate limit and latency
    tc qdisc add dev $IFACE root handle 1: htb default 10
    tc class add dev $IFACE parent 1: classid 1:10 htb rate 10gbit
    tc qdisc add dev $IFACE parent 1:10 handle 10: netem delay 0.05ms

    echo "✓ LAN emulation configured"
    echo ""
    echo "Verification:"
    tc qdisc show dev $IFACE
}

# Function to setup WAN emulation
setup_wan() {
    echo "Setting up WAN emulation on $IFACE..."
    echo "  - Bandwidth: 100 Mbps"
    echo "  - RTT: 50 ms (latency: 25 ms)"
    echo "  - No packet loss"

    # Reset first
    reset_network

    # Add qdisc with rate limit and latency
    tc qdisc add dev $IFACE root handle 1: htb default 10
    tc class add dev $IFACE parent 1: classid 1:10 htb rate 100mbit
    tc qdisc add dev $IFACE parent 1:10 handle 10: netem delay 25ms

    echo "✓ WAN emulation configured"
    echo ""
    echo "Verification:"
    tc qdisc show dev $IFACE
}

# Function to verify current settings
verify_settings() {
    echo "Current network emulation on $IFACE:"
    echo ""
    tc qdisc show dev $IFACE
}

# Main
case "${1:-}" in
    lan)
        setup_lan
        ;;
    wan)
        setup_wan
        ;;
    reset)
        reset_network
        ;;
    verify)
        verify_settings
        ;;
    *)
        echo "Usage: sudo $0 {lan|wan|reset|verify}"
        echo ""
        echo "Commands:"
        echo "  lan    - Setup LAN emulation (10 Gbps, 0.1 ms RTT)"
        echo "  wan    - Setup WAN emulation (100 Mbps, 50 ms RTT)"
        echo "  reset  - Reset network emulation"
        echo "  verify - Show current settings"
        echo ""
        echo "Example:"
        echo "  sudo $0 lan"
        echo "  ./benchmark_table3_real_network"
        echo "  sudo $0 reset"
        exit 1
        ;;
esac
