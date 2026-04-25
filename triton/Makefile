.PHONY: help profile-rms

help:
	@echo "Targets:"
	@echo "  profile-rms   Profile Triton rms_norm_kernel with ncu"

profile-rms:
	mkdir -p ncu/
	ncu -k rms_norm_kernel python rms_norm.py 2>&1 > ncu/rms_norm.txt

