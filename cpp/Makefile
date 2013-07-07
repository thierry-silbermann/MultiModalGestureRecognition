# This makefile is on purpose not made with configure, to show how to use the library
# The make file requires that the fann library is installed (see ../README)

GCC=gcc

TARGETS = sound_training sound_validation

all: $(TARGETS)

%: %.c Makefile
	$(GCC) -O3 $< -o $@ -lfann -lm

%_fixed: %.c Makefile
	$(GCC) -O3 -DFIXEDFANN $< -o $@ -lfixedfann -lm

clean:
	rm -f $(TARGETS) *.net *~ *.obj *.exe *.tds

