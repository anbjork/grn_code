"""
Copyright (C) 2022  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Extracted this from the PSGRN repo
https://github.com/GuanLab/PSGRN
, to make this repo self contained, loose all the CausalBench challenge specific wrapper code, and get PSGRN as a standalone inference method. I've made some changes (but not to this specific file), just for information. //AB
"""

import enum


class TrainingRegime(enum.Enum):
    Observational = "observational"
    PartialIntervational = "partial_interventional"
    Interventional = "interventional"

