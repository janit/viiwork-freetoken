package gpu

import (
	"strconv"
	"strings"
	"time"
)

// notAvailable is what nvidia-smi prints for a field the driver cannot supply.
// It appears per-field rather than per-row, so a card with no power sensor
// still reports real utilisation and VRAM on the same line.
const notAvailable = "[N/A]"

// parseCSV turns `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
// output into samples.
//
// Rows are parsed independently and a malformed one is skipped rather than
// failing the batch: a driver that cannot answer for one card must not cost
// us the readings for the other nine on the host. The same reasoning applies
// per field — an unparseable power figure leaves PowerW at zero (reported as
// absent) while utilisation and VRAM from the same row are kept.
func parseCSV(out []byte, now time.Time) []GPUSample {
	var samples []GPUSample
	ts := now.Unix()
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		// index, utilization.gpu, memory.used, memory.total are required; a
		// row without them says nothing worth keeping.
		if len(fields) < 4 {
			continue
		}
		id, err := strconv.Atoi(fields[0])
		if err != nil {
			continue
		}
		total, err := parseFloat(fields[3])
		if err != nil {
			continue
		}
		s := GPUSample{GPUID: id, VRAMTotalMB: total, Timestamp: ts}
		// Utilisation is [N/A] on some virtualised and MIG configurations.
		// Zero is the honest fallback: the card is visible, we cannot see load
		// on it, and the alternative is dropping the card from the fleet view
		// entirely.
		if v, err := parseFloat(fields[1]); err == nil {
			s.Utilization = v
		}
		if v, err := parseFloat(fields[2]); err == nil {
			s.VRAMUsedMB = v
		}
		if len(fields) >= 5 {
			if v, err := parseFloat(fields[4]); err == nil {
				s.PowerW = v
			}
		}
		samples = append(samples, s)
	}
	return samples
}

func parseFloat(s string) (float64, error) {
	if s == "" || s == notAvailable || strings.HasPrefix(s, "[") {
		return 0, strconv.ErrSyntax
	}
	return strconv.ParseFloat(s, 64)
}

// hasPower reports whether any row carried a real wattage. Judged by parsing
// the output rather than by the command's exit status, because nvidia-smi
// accepts power.draw on hardware that then reports [N/A] for every card —
// exit 0, no data. Energy attribution needs to know the difference.
func hasPower(samples []GPUSample) bool {
	for _, s := range samples {
		if s.PowerW > 0 {
			return true
		}
	}
	return false
}
