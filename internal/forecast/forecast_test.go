package forecast

import "testing"

func TestRun(t *testing.T) {
	resp, err := Run(Request{Series: []float64{1, 2, 3, 4, 5}, Horizon: 2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Method != "moving_average" {
		t.Fatalf("expected method moving_average, got %q", resp.Method)
	}
	if len(resp.Forecast) != 2 {
		t.Fatalf("expected 2 forecast points, got %d", len(resp.Forecast))
	}
	if resp.Forecast[0] != 3 {
		t.Fatalf("expected first point to be the mean of [1..5] = 3, got %v", resp.Forecast[0])
	}
}

func TestRunEmptySeries(t *testing.T) {
	if _, err := Run(Request{Series: nil, Horizon: 1}); err == nil {
		t.Fatal("expected error for empty series")
	}
}

func TestRunInvalidHorizon(t *testing.T) {
	if _, err := Run(Request{Series: []float64{1, 2, 3}, Horizon: 0}); err == nil {
		t.Fatal("expected error for non-positive horizon")
	}
}
