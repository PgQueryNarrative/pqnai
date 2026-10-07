package pgjobs

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Job struct {
	ID      int64
	Type    string
	Payload []byte
}

// ClaimNext locks and claims the oldest queued job, if any, using
// SKIP LOCKED so multiple worker instances can run concurrently.
func ClaimNext(ctx context.Context, pool *pgxpool.Pool) (*Job, bool, error) {
	row := pool.QueryRow(ctx, `
		UPDATE pqnai.jobs
		SET status = 'running', updated_at = now()
		WHERE id = (
			SELECT id FROM pqnai.jobs
			WHERE status = 'queued'
			ORDER BY id
			FOR UPDATE SKIP LOCKED
			LIMIT 1
		)
		RETURNING id, job_type, payload
	`)

	var job Job
	if err := row.Scan(&job.ID, &job.Type, &job.Payload); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, false, nil
		}
		return nil, false, err
	}
	return &job, true, nil
}

func Complete(ctx context.Context, pool *pgxpool.Pool, id int64, result []byte) error {
	_, err := pool.Exec(ctx, `
		UPDATE pqnai.jobs SET status = 'done', result = $1, updated_at = now() WHERE id = $2
	`, result, id)
	return err
}

func Fail(ctx context.Context, pool *pgxpool.Pool, id int64, errMsg string) error {
	_, err := pool.Exec(ctx, `
		UPDATE pqnai.jobs SET status = 'failed', error = $1, updated_at = now() WHERE id = $2
	`, errMsg, id)
	return err
}
