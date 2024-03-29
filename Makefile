# BSD 3-Clause License

# Copyright (c) 2020, Juan L Gamella
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SUITE = all
PROJECT = causalicp
CASES_DIR = causalicp/test/test_cases

# Run tests
tests: test doctests examples

test: cases
ifeq ($(SUITE),all)
	python -m unittest discover $(PROJECT).test
else
	python -m unittest $(PROJECT).test.$(SUITE)
endif

# Generate random cases for comparison with the R implementation, but
# don't overwrite them if they're already there!
NO_CASES = 1000

cases: $(CASES_DIR)

$(CASES_DIR):
	rm -rf $(CASES_DIR)
	mkdir $(CASES_DIR)
	PYTHONPATH=./ python causalicp/test/generate_test_cases.py --G $(NO_CASES)
	Rscript causalicp/test/run_icp.R $(NO_CASES)

# Run the doctests
doctests:
	PYTHONPATH=./ python causalicp/icp.py
	PYTHONPATH=./ python causalicp/data.py

# Run the example scripts in the README
examples:
	PYTHONPATH=./ python docs/example.py
	PYTHONPATH=./ python docs/example_w_sempler.py

clean:
	rm -rf $(CASES_DIR)

.PHONY: test, tests, examples
