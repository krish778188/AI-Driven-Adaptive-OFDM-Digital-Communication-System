"""Unit tests for simulation interface and JSON/CSV data exchange."""

import os
import pytest
from python_src.simulation_interface import SimulationInterface


def test_interface_initialization():
    interface = SimulationInterface(workspace_dir="D:\\DC project")
    status = interface.get_engine_status()

    assert "matlab_engine_installed" in status
    assert status["fallback_engine_available"] is True
    assert len(status["integration_modes"]) == 3


def test_single_point_simulation_execution():
    interface = SimulationInterface(workspace_dir="D:\\DC project")
    cfg = {
        "mod_type": "QPSK",
        "snr_db": 15.0,
        "channel_type": "AWGN",
        "num_bits": 4800,
        "target_ber": 0.01,
    }
    res = interface.run_simulation(cfg)

    assert res["mod_type"] == "QPSK"
    assert res["ber"] <= 0.01
    assert "net_throughput_mbps" in res
    assert res["execution_time_sec"] > 0.0


def test_config_json_export():
    interface = SimulationInterface(workspace_dir="D:\\DC project")
    cfg = {"mod_type": "16-QAM", "snr_db": 18.0}
    json_path = interface.export_config_for_matlab(cfg, filename="test_export.json")

    assert os.path.exists(json_path)
    os.remove(json_path)
