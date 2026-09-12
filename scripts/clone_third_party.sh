#!/usr/bin/env bash
# clone VCD and the VLMBias generators at the commits I used (see VERSIONS.md)
set -euo pipefail
cd "$(dirname "$0")/../third_party"
[ -d VCD ] || git clone https://github.com/DAMO-NLP-SG/VCD.git
git -C VCD checkout -q d6568ff81b8fd306a49e630df44f2db5c2300191
[ -d vlms-are-biased ] || git clone https://github.com/anvo25/vlms-are-biased.git
git -C vlms-are-biased checkout -q c8aaa69c71c66ed3883ce69c1e858f3862f1e5c9
