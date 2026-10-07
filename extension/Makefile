MODULES = pqnai
EXTENSION = pqnai
DATA = pqnai--0.1.0.sql
PGFILEDESC = "pqnai - AI forecasting and retrieval for Postgres"

REGRESS = basic
REGRESS_OPTS = --inputdir=../test/regress --outputdir=../test/regress

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
