MODULES = pqnai
EXTENSION = pqnai
DATA = pqnai--0.1.0.sql pqnai--0.1.0--0.2.0.sql pqnai--0.2.0.sql pqnai--0.2.0--0.3.0.sql pqnai--0.3.0.sql pqnai--0.3.0--0.4.0.sql pqnai--0.4.0.sql pqnai--0.4.0--0.5.0.sql pqnai--0.5.0.sql pqnai--0.5.0--0.6.0.sql pqnai--0.6.0.sql
PGFILEDESC = "pqnai - AI forecasting and retrieval for Postgres"

REGRESS = basic rag
REGRESS_OPTS = --inputdir=../test/regress --outputdir=../test/regress

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
