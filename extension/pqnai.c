#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pqnai_version);

/*
 * Looks up the actually-installed extension version via pg_extension
 * rather than returning a hardcoded string, so this can't silently drift
 * out of sync with pqnai.control on a version bump.
 */
Datum
pqnai_version(PG_FUNCTION_ARGS)
{
	int			ret;
	text	   *result;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "pqnai_version: SPI_connect failed");

	ret = SPI_execute(
		"SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'pqnai'",
		true, 1);

	if (ret != SPI_OK_SELECT || SPI_processed != 1)
	{
		SPI_finish();
		elog(ERROR, "pqnai_version: could not find pqnai in pg_extension");
	}

	result = cstring_to_text(
		SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1));

	SPI_finish();

	PG_RETURN_TEXT_P(result);
}
