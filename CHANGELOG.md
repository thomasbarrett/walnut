# Changelog

## [1.0.0](https://github.com/thomasbarrett/walnut/compare/v0.3.0...v1.0.0) (2026-08-21)


### ⚠ BREAKING CHANGES

* **bench:** replace the skill script with a `walnut bench` command ([#52](https://github.com/thomasbarrett/walnut/issues/52))
* serving requires a CUDA device of compute capability 8.0 (Ampere) or newer, in float16 or bfloat16. `--device cpu` and `--dtype float32` are now rejected at startup rather than served slowly. The `cpu` extra still installs a torch that lints, type-checks and runs the test suite, which is what CI uses, but it cannot serve a model. `Attention.forward` and the model `forward` methods now require a cache rather than accepting `None`.
* walnut_chat_completions_total is removed. The request duration histogram's _count series replaces it, without the `stream` label.

### Features

* add benchmark and optimize skills ([#38](https://github.com/thomasbarrett/walnut/issues/38)) ([0960095](https://github.com/thomasbarrett/walnut/commit/0960095f595f721f5958ab571525a53b6abc373b))
* analyze profiler traces with an analyze-trace skill ([#32](https://github.com/thomasbarrett/walnut/issues/32)) ([0236bf1](https://github.com/thomasbarrett/walnut/commit/0236bf1dd92dfeaa77c4efaa6992b60330788bf6))
* **bench:** make workload shape the primary axis ([#59](https://github.com/thomasbarrett/walnut/issues/59)) ([a953441](https://github.com/thomasbarrett/walnut/commit/a9534415321b3fcb0a595379975948f7cf92003a))
* **bench:** replace the skill script with a `walnut bench` command ([#52](https://github.com/thomasbarrett/walnut/issues/52)) ([2b6e29c](https://github.com/thomasbarrett/walnut/commit/2b6e29cc5b6c27c48d8a83dcd816b5c43d6a7906))
* **bench:** report throughput at an SLO, on every mode ([#58](https://github.com/thomasbarrett/walnut/issues/58)) ([4ad4cd9](https://github.com/thomasbarrett/walnut/commit/4ad4cd943138794c3dcbf856e924a9b9901be985))
* **bench:** sweep a concurrency ladder, and draw the frontier ([#57](https://github.com/thomasbarrett/walnut/issues/57)) ([aaba6b3](https://github.com/thomasbarrett/walnut/commit/aaba6b3376dfd2678c42e5183435148f4dfa2b51))
* **cache:** page the KV cache, ending the slot-per-request reservation ([#63](https://github.com/thomasbarrett/walnut/issues/63)) ([fcab4d5](https://github.com/thomasbarrett/walnut/commit/fcab4d546f91a8c55f1d723557eddcd098e96809))
* decode a continuous batch of requests, sized by --max-batch-size ([#51](https://github.com/thomasbarrett/walnut/issues/51)) ([b68608f](https://github.com/thomasbarrett/walnut/commit/b68608f4f8c6635fc17b1931b61ba1652648e3f9))
* load models the same way in profile as in serve ([#31](https://github.com/thomasbarrett/walnut/issues/31)) ([468c97d](https://github.com/thomasbarrett/walnut/commit/468c97df2268aa48496719bebf8ef776dcbb59bf))
* profile with torch.profiler and export a Chrome trace ([#29](https://github.com/thomasbarrett/walnut/issues/29)) ([e5b2b3c](https://github.com/thomasbarrett/walnut/commit/e5b2b3cc700afeacba1c8ceb1a5c899a23aa90f3))
* replay the decode step from a CUDA graph ([#27](https://github.com/thomasbarrett/walnut/issues/27)) ([a7edec6](https://github.com/thomasbarrett/walnut/commit/a7edec6e484ddf9d4e55eb52ac22dd65f4a7dd99))
* report GenAI semantic convention metrics ([#28](https://github.com/thomasbarrett/walnut/issues/28)) ([7c8dcd9](https://github.com/thomasbarrett/walnut/commit/7c8dcd9d3711b7b078bf9b5da5414851eb516e84))
* select device and dtype for serving ([70cff36](https://github.com/thomasbarrett/walnut/commit/70cff36d03382c7c8ddf5f0a4c3716e655d12d44))
* select device and dtype for serving ([fdc9645](https://github.com/thomasbarrett/walnut/commit/fdc96455076c68617a54e9ba7e67817ecd305cae))
* static pre-allocated KV cache ([8a7aac0](https://github.com/thomasbarrett/walnut/commit/8a7aac0e42fcaed12bfb486b009585c55c8e2c52))
* static pre-allocated KV cache ([d1ee09f](https://github.com/thomasbarrett/walnut/commit/d1ee09f0a048311752a8fc12eb547aef5538ef04))


### Bug Fixes

* **benchmark:** tell a cold-cache compile apart from a first-request regression ([#42](https://github.com/thomasbarrett/walnut/issues/42)) ([7547d26](https://github.com/thomasbarrett/walnut/commit/7547d265a9eaa5746a41fd1badda8464398d2205))
* load vision MLP weights instead of silently randomizing them ([e8964d0](https://github.com/thomasbarrett/walnut/commit/e8964d0df3fae8c407e60a3738ddbb82bc085490))
* load vision MLP weights instead of silently randomizing them ([3285d53](https://github.com/thomasbarrett/walnut/commit/3285d531469786760f591e01de8845d2538dc0a9))


### Performance Improvements

* autotune the decode projections, cutting TPOT 11% ([#48](https://github.com/thomasbarrett/walnut/issues/48)) ([8e04a6e](https://github.com/thomasbarrett/walnut/commit/8e04a6e5a29ea0f3928a9c3eceeecb0e14e7d1bd))
* chunk prefill through the scheduler, cutting agentic ITL p99 84% ([#60](https://github.com/thomasbarrett/walnut/issues/60)) ([dc46410](https://github.com/thomasbarrett/walnut/commit/dc464108e720225e0a501f02b13c7ed09ca049f6))
* chunk prefill's delta rule and stop holding the first token, cutting TTFT 58% ([#49](https://github.com/thomasbarrett/walnut/issues/49)) ([a73d438](https://github.com/thomasbarrett/walnut/commit/a73d43801dafaa98bd65041c512e77123cebf091))
* compile the decode step, halving time per output token ([#37](https://github.com/thomasbarrett/walnut/issues/37)) ([56eef23](https://github.com/thomasbarrett/walnut/commit/56eef231a96db88c7065a287a087c12132327dcb))
* fuse the projections that share an input, cutting TPOT 9% ([#40](https://github.com/thomasbarrett/walnut/issues/40)) ([d16047d](https://github.com/thomasbarrett/walnut/commit/d16047d8b8ffbcd1755f8d6f022d71e7830ce597))
* widen the delta rule's prefill chunk, doubling throughput at 1k prompts ([#54](https://github.com/thomasbarrett/walnut/issues/54)) ([62f8672](https://github.com/thomasbarrett/walnut/commit/62f867226bbbc980c10ec1d9b5926d758ac24468))


### Documentation

* close the gaps a trace-analysis run exposed in analyze-trace ([#35](https://github.com/thomasbarrett/walnut/issues/35)) ([5c9c14d](https://github.com/thomasbarrett/walnut/commit/5c9c14d1397acbe9e0e4e6d280a0250dbb7eb9af))
* cut duplicated and unreachable material from analyze-trace ([#36](https://github.com/thomasbarrett/walnut/issues/36)) ([f4a48df](https://github.com/thomasbarrett/walnut/commit/f4a48dff52bbb3abeced4df8c7f63feba005c857))
* make the hook config the single definition of the checks ([#44](https://github.com/thomasbarrett/walnut/issues/44)) ([261a542](https://github.com/thomasbarrett/walnut/commit/261a54222b4a2adfac23ef546461c1f1b2531e68))

## [0.3.0](https://github.com/thomasbarrett/walnut/compare/v0.2.1...v0.3.0) (2026-07-14)


### Features

* add Qwen3.5 model support ([02d631e](https://github.com/thomasbarrett/walnut/commit/02d631efee5e1b5fa6b4ae00419a5bb4d86f006a))
* add Qwen3.5 model support ([cb9fd59](https://github.com/thomasbarrett/walnut/commit/cb9fd59a9df1c25cdc636b4b85ae08a4a04f328f))

## [0.2.1](https://github.com/thomasbarrett/walnut/compare/v0.2.0...v0.2.1) (2026-07-13)


### Bug Fixes

* drop uvicorn color_message from structured logs ([d2140f3](https://github.com/thomasbarrett/walnut/commit/d2140f37af047f46e219cf58f4ec3297fe8b1af8))
* drop uvicorn color_message from structured logs ([d23463f](https://github.com/thomasbarrett/walnut/commit/d23463f9b8f04817b2dadd0b8acb546f56fe1d92))

## [0.2.0](https://github.com/thomasbarrett/walnut/compare/v0.1.0...v0.2.0) (2026-07-13)


### Features

* initial walnut inference server ([0165edf](https://github.com/thomasbarrett/walnut/commit/0165edf9e601e87ab1ac72d3b6c67406bd9f1ec4))
