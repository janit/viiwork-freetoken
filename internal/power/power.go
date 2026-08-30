// Package power reports node wattage on a host with no BMC, by summing the
// per-GPU board power nvidia-smi already collects.
//
// This is the one place the two implementations of the mesh genuinely disagree
// about what a number means. viiwork measures the chassis over IPMI or DCMI:
// CPU, fans, drives and PSU losses included, around 400W on an idle 10-GPU
// host. There is no uniform BMC across the NVIDIA hosts, so this node publishes
// the sum of the cards instead, which excludes all of that and understates the
// wall figure substantially.
//
// That is a deliberate trade, not an approximation to be improved by guessing:
// an operator-supplied overhead constant would put an estimate into a field
// that is measured everywhere else in the fleet, and a wrong guess becomes
// invisible the moment it is summed into a fleet total. What makes it honest
// instead is the label — Source reports "nvidia-smi", which travels on
// meshapi's PowerSource and is recorded in the energy store's directory, so
// neither a dashboard nor a history copied off the host can mistake it for a
// chassis reading.
package power

import (
	"github.com/janit/viiwork/energy"

	"github.com/janit/viiwork-freetoken/internal/gpu"
)

// SourceLabel is the provenance label for a summed per-GPU reading. It uses the
// same vocabulary as meshapi's PowerSource field and energy.Config.Source,
// where viiwork writes "dcmi", "sdr" or "sensor:<NAME>" for a chassis reading.
const SourceLabel = "nvidia-smi"

// Sum is node wattage as the total of every card's board power.
//
// It satisfies peer.PowerReader and energy.NodeWattsFunc, and is deliberately
// backed by the same gpu.History the dashboard reads: the node figure and the
// per-GPU readings that Direct attribution divides it into must come from one
// set of samples, or the two series stop reconciling.
type Sum struct {
	history *gpu.History
	// powerAvailable is the collector's own judgement, kept as a function
	// rather than a bool because it flips at runtime — a driver that starts
	// reporting "[N/A]" is a live condition, not a startup property.
	powerAvailable func() bool
}

// NewSum builds a node-wattage source over a GPU history. powerAvailable is
// normally StatCollector.PowerAvailable; a nil one reads as "no usable
// wattage", which is the safe direction.
func NewSum(history *gpu.History, powerAvailable func() bool) *Sum {
	return &Sum{history: history, powerAvailable: powerAvailable}
}

// Watts is the summed board power of every GPU the host reports.
//
// It spans the whole host rather than this instance's configured slice, for the
// same reason gpu.History does: node wattage is a per-host measurement, and a
// sum over one instance's cards would be neither the node's draw nor a
// meaningful share of it. This is also why the energy recorder must run on
// exactly one instance per host.
//
// Zero when wattage is unavailable, so meshapi's omitempty drops the field
// rather than publishing a confident nought. A card that is powered on never
// draws 0W, so absent has to stay distinguishable from measured.
func (s *Sum) Watts() float64 {
	if !s.Available() {
		return 0
	}
	var total float64
	for _, sample := range s.history.Latest() {
		total += sample.PowerW
	}
	return total
}

// Available reports whether any wattage is being measured at all.
func (s *Sum) Available() bool {
	if s.powerAvailable == nil {
		return false
	}
	return s.powerAvailable()
}

// Source names what Watts measured, for meshapi's PowerSource.
func (s *Sum) Source() string { return SourceLabel }

// NodeWatts adapts Sum to energy.NodeWattsFunc.
func (s *Sum) NodeWatts() (float64, bool) { return s.Watts(), s.Available() }

// Readings is the per-GPU half of the same observation: each card's measured
// draw, stamped with whichever model is resident on it.
//
// models maps GPU id to model name and is expected to cover co-located
// instances as well as this one — see peer.Registry.GPUModels. A card no model
// claims is reported with an empty name rather than dropped: it still draws
// power that belongs in the node total, and the store treats the empty name as
// unattributed.
//
// The readings sum to Sum.Watts by construction, both being derived from
// gpu.History.Latest. energy.Direct relies on that: it attributes each card its
// own draw, so the shares reconcile with the node figure exactly and the
// baseline is honestly zero. See energy.Direct for why the chassis model's
// inferred baseline is wrong for this producer.
func Readings(history *gpu.History, models map[int]string) []energy.GPUReading {
	latest := history.Latest()
	out := make([]energy.GPUReading, 0, len(latest))
	for _, sample := range latest {
		out = append(out, energy.GPUReading{
			GPUID: sample.GPUID,
			Watts: sample.PowerW,
			Model: models[sample.GPUID],
		})
	}
	return out
}
