// Package forecast implements time-series forecasting handlers for pqnai jobs.
//
// v0.1 ships a simple moving-average forecaster to prove the end-to-end
// job queue round trip. ARIMA/ETS implementations land in a later phase.
package forecast

import "fmt"

type Request struct {
	Series  []float64 `json:"series"`
	Horizon int       `json:"horizon"`
}

type Response struct {
	Forecast []float64 `json:"forecast"`
	Method   string    `json:"method"`
}

const movingAverageWindow = 5

// Run produces a Horizon-step-ahead forecast using a recursive simple
// moving average: each new point is the average of the trailing window,
// and the window slides forward to include previously forecast points.
func Run(req Request) (Response, error) {
	if len(req.Series) == 0 {
		return Response{}, fmt.Errorf("forecast: series must not be empty")
	}
	if req.Horizon <= 0 {
		return Response{}, fmt.Errorf("forecast: horizon must be positive, got %d", req.Horizon)
	}

	series := append([]float64(nil), req.Series...)
	window := movingAverageWindow
	if window > len(series) {
		window = len(series)
	}

	forecast := make([]float64, 0, req.Horizon)
	for i := 0; i < req.Horizon; i++ {
		sum := 0.0
		for _, v := range series[len(series)-window:] {
			sum += v
		}
		next := sum / float64(window)
		forecast = append(forecast, next)
		series = append(series, next)
	}

	return Response{Forecast: forecast, Method: "moving_average"}, nil
}
