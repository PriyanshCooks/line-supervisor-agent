PY ?= python
.PHONY: ingest sample vision serve app test eval report
ingest:  ; $(PY) tasks.py ingest
sample:  ; $(PY) tasks.py sample
vision:  ; $(PY) tasks.py vision
serve:   ; $(PY) tasks.py serve
app:     ; $(PY) tasks.py app
test:    ; $(PY) tasks.py test
eval:    ; $(PY) tasks.py eval
report:  ; $(PY) tasks.py report
