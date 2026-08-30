package config

import "testing"

// The shipped example must load. It is the first thing anyone copies, and a
// key renamed in the code but not in the example is a broken quick start.
func TestExampleConfigLoads(t *testing.T) {
	cfg, err := Load("../../viiwork-freetoken.yaml.example")
	if err != nil {
		t.Fatalf("viiwork-freetoken.yaml.example does not load: %v", err)
	}
	if cfg.FreeToken.Binary != "ft" || cfg.FreeToken.MaxRunningRequests != 4 || cfg.FreeToken.MemoryRatio != 0.90 {
		t.Errorf("example does not set the freetoken block it documents: %+v", cfg.FreeToken)
	}
	if cfg.Server.Port != DefaultPort || cfg.GPUs.BasePort != DefaultBasePort {
		t.Errorf("example ports drifted from the defaults: server=%d base=%d", cfg.Server.Port, cfg.GPUs.BasePort)
	}
	if len(cfg.Backends()) != len(cfg.GPUs.Devices) {
		t.Error("one backend per device")
	}
}
