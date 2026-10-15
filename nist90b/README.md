# NIST SP 800-90B recomputation

Build the official source at commit `68ed165fd7a3eeef26b87a546ba23f338e82a3f3` (tag `v1.1.8`):

```sh
sudo apt-get install build-essential libbz2-dev libdivsufsort-dev libjsoncpp-dev libssl-dev
git clone https://github.com/usnistgov/SP800-90B_EntropyAssessment.git
cd SP800-90B_EntropyAssessment
git checkout 68ed165fd7a3eeef26b87a546ba23f338e82a3f3
cd cpp
make -j4 iid non_iid
```

The commit pins the validated implementation; `shared/utils.h` reports version **1.1.7** at this tag. The official source and its software notice are available in the upstream repository.

From the artifact root, with the Python requirements installed:

```sh
python nist90b/recompute.py --tools /path/to/SP800-90B_EntropyAssessment/cpp
```

The runner creates one-byte-per-symbol binary inputs and keeps complete tool logs, input/executable SHA-256 hashes, arguments, and parsed estimates in `data/recomputed/`. Spectre inputs use voltage >= 0.5 as one, exactly as Figure 8. ASIC inputs use the lowest bit of each byte, and the same eight 27 C minimum-bias captures as Table 2. All available samples are assessed (`-i -a -v`, one bit per symbol) as raw, unconditioned outputs. Both IID and non-IID tracks run for Table 1, and non-IID for Table 2. Read the entropy estimates together with the IID-test decisions recorded in the logs. Permutation-test counts can vary with the random permutations.

The default table commands use archived NIST estimates. Fresh estimates and logs are saved under `nist90b/data/recomputed/`. Verify them with:

```sh
python nist90b/verify.py
```

The Table 2 1.1 V/F3 and 1.3 V/F3 captures contain 984733 and 984750 samples, respectively; the NIST tool reports a below-one-million sample warning.
