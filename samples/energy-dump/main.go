// Command energy-dump prints what a durable energy store contains.
//
// The mesh publishes only whole-node energy_kwh_24h and energy_kwh_30d. The
// per-GPU and per-model detail is recorded but never crosses the wire, in
// either implementation, so this is how you read it.
//
// The on-disk format belongs to viiwork and is shared, so this works against a
// store written by either implementation — an NVIDIA node summing per-GPU board
// power, or an AMD one measuring the chassis over IPMI. Store.Source says which,
// and it matters: the same NodeRecord.Watts field means whole-chassis draw in
// one case and GPU-only in the other.
//
// It also checks the two invariants a direct-measurement producer must satisfy,
// which nothing else verifies:
//
//   - AttrW == RawW on every GPU record. A producer whose node figure is the sum
//     of its own per-GPU readings has no aggregate to divide, so each card is
//     charged its own measured draw.
//   - The shares reconcile: sum(AttrW) == NodeRecord.Watts for each bucket, so
//     the baseline is zero rather than absorbing real load.
//
// A store written by a chassis-measuring node will legitimately fail both — its
// AttrW is a marginal share and its baseline is real overhead — so the check is
// reported, not enforced, and interpreted against Source.
//
// Usage:
//
//	energy-dump <store-dir>
//
// Run it against a COPY. A store has one recorder, and pointing this at a live
// directory means two processes hold the same files.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"time"

	"github.com/janit/viiwork/energy"
)

func main() {
	window := flag.Duration("window", 24*time.Hour, "how far back to read")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: energy-dump [-window 24h] <store-dir>")
		os.Exit(2)
	}
	dir := flag.Arg(0)

	// GPUIDs must cover the lanes the store already has; a shorter list would
	// not match its geometry. Source is left empty on purpose: that leaves
	// whatever label the directory carries untouched rather than relabelling
	// somebody else's history as ours.
	ids, err := laneIDs(dir)
	if err != nil {
		log.Fatalf("reading store geometry: %v", err)
	}
	store, err := energy.Open(energy.Config{Dir: dir, GPUIDs: ids}, log.New(os.Stderr, "", 0))
	if err != nil {
		log.Fatalf("opening store: %v", err)
	}
	defer store.Close()

	src := store.Source()
	if src == "" {
		// Absent means unknown, never a default. Stores written before the
		// label existed have none.
		src = "(unrecorded)"
	}
	now := time.Now()
	from := now.Add(-*window)

	fmt.Printf("store   : %s\n", dir)
	fmt.Printf("source  : %s\n", src)
	fmt.Printf("node    : %.6f kWh (24h)   %.6f kWh (30d)\n", store.KWh24h(), store.KWh30d())
	fmt.Printf("on disk : %d bytes\n\n", store.DiskBytes())

	nodes := store.ReadNode(energy.TierMinute, from, now)
	gpus := store.ReadGPU(energy.TierMinute, from, now)
	fmt.Printf("minute records in the last %s: %d node, %d gpu\n", *window, len(nodes), len(gpus))

	nodeAt := make(map[int64]energy.NodeRecord, len(nodes))
	for _, n := range nodes {
		nodeAt[n.TS] = n
	}

	attrAt := make(map[int64]float64, len(nodes))
	var mismatched, buckets, reconciled int
	for _, g := range gpus {
		attrAt[g.TS] += float64(g.AttrW)
		if g.AttrW != g.RawW {
			mismatched++
		}
	}

	if len(nodes) > 0 {
		fmt.Println("\nlatest buckets:")
		fmt.Println("  time      nodeW  covered  kWh          sum(attrW)  reconciles")
		for _, n := range tail(nodes, 10) {
			sum := attrAt[n.TS]
			ok := within(sum-float64(n.Watts), 0.01)
			fmt.Printf("  %s  %6.2f  %5ds  %.9f  %10.2f  %v\n",
				time.Unix(n.TS, 0).Format("15:04:05"), n.Watts, n.CoveredS, n.KWh(), sum, ok)
		}
	}
	for ts, sum := range attrAt {
		if n, ok := nodeAt[ts]; ok {
			buckets++
			if within(sum-float64(n.Watts), 0.01) {
				reconciled++
			}
		}
	}

	byModel := store.ByModel(energy.TierMinute, from, now)
	if len(byModel) > 0 {
		fmt.Println("\nper model:")
		names := make([]string, 0, len(byModel))
		for m := range byModel {
			names = append(names, m)
		}
		sort.Strings(names)
		for _, m := range names {
			label := m
			if label == "" {
				// A card no model claimed. Its draw still belongs in the node
				// total, so it is reported rather than dropped.
				label = "(unattributed)"
			}
			fmt.Printf("  %-32s %.9f kWh\n", label, byModel[m])
		}
	}

	fmt.Printf("\ndirect-attribution check: %d/%d gpu records have AttrW == RawW, %d/%d buckets reconcile\n",
		len(gpus)-mismatched, len(gpus), reconciled, buckets)
	if len(gpus) > 0 && mismatched == 0 && buckets > 0 && reconciled == buckets {
		fmt.Println("  consistent with a direct (summed per-GPU) producer")
	} else if len(gpus) > 0 {
		fmt.Println("  consistent with a chassis producer: AttrW is a marginal share and the residual is baseline")
	}
}

// laneIDs recovers the GPU ids a store was created with, so it can be reopened
// without guessing its geometry. The model table is not enough — it names
// models, not cards — so this reads the ids off the existing GPU records.
func laneIDs(dir string) ([]int, error) {
	probe, err := energy.Open(energy.Config{Dir: dir, GPUIDs: []int{0}}, log.New(os.Stderr, "", 0))
	if err != nil {
		return nil, err
	}
	defer probe.Close()
	seen := map[int]bool{}
	for _, g := range probe.ReadGPU(energy.TierDay, time.Now().Add(-365*24*time.Hour), time.Now()) {
		seen[int(g.GPUID)] = true
	}
	if len(seen) == 0 {
		return []int{0}, nil
	}
	ids := make([]int, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids, nil
}

func within(v, eps float64) bool { return v < eps && v > -eps }

func tail[T any](s []T, n int) []T {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
