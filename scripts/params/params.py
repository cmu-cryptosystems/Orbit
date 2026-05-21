import json
import math
from ..resilience import ResilienceProfile

DEFAULT_OPENEVOLVE_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENEVOLVE_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_OPENEVOLVE_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENEVOLVE_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENEVOLVE_LLM_MAX_TOKENS = 50_000

DEFAULT_RESILIENCE_ERROR_MODEL = {
    "input_error_abs": 0.0,
    "add_error_abs": 1e-10,
    "mul_plain_error_abs": 1e-8,
    "mul_cipher_error_abs": 2e-8,
    "rotate_error_abs": 1e-9,
    "rescale_error_abs": 1e-8,
    "modswitch_error_abs": 1e-9,
    "upscale_error_abs": 0.0,
    "bootstrap_error_abs": 1e-6,
    "max_error_abs": 1.0,
}


def _parse_scale_floor_candidates(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


class Params:
    def __init__(
        self,
        le_json,
        sysname,
        mode,
        Sw=None,
        CSw=None,
        bpsdepth=None,
        threads=None,
        comp=None,
        part=None,
        reqbp=None,
        netname=None,
        ilp_solver=None,
        placement_backend=None,
        resilience_profile=None,
        resilience_mode="waterline",
        allow_empty_resilience_match=False,
        resilience_decomposition="off",
        resilience_decompose_threshold=32,
        resilience_max_boundary_states=8,
        resilience_error_buckets=8,
        ilp_task_time_limit_sec=0.0,
        resilience_constraint_policy="relax-only",
        openevolve_config=None,
        openevolve_output_dir=None,
        openevolve_iterations=0,
        openevolve_seed=42,
        openevolve_keep_workdir=False,
        openevolve_provider=None,
        openevolve_model=None,
        openevolve_api_base=None,
        openevolve_api_key_env=None,
        openevolve_primary_weight=None,
        openevolve_secondary_provider=None,
        openevolve_secondary_model=None,
        openevolve_secondary_api_base=None,
        openevolve_secondary_api_key_env=None,
        openevolve_secondary_weight=None,
        openevolve_harness="compile",
        openevolve_search_mode=None,
        openevolve_granularity=None,
        openevolve_leniency=None,
        openevolve_max_unit_samples=None,
        openevolve_eval_suite="polybert-sampled",
        openevolve_reference_json=None,
        openevolve_finalists=3,
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=0,
        openevolve_llm_timeout_sec=180,
        openevolve_llm_retries=1,
        openevolve_llm_retry_delay_sec=2,
        openevolve_llm_max_tokens=None,
        openevolve_evaluator_timeout_sec=180,
        openevolve_parallel_evaluations=1,
        openevolve_checkpoint_interval=5,
        openevolve_fail_open=True,
        openevolve_reuse_output=False,
        noise_estimator="finalists",
        noise_estimator_binary=None,
        noise_estimator_timeout_sec=30,
        noise_estimator_min_output_margin_bits=2.0,
        noise_estimator_alpha=14.0,
        noise_estimator_max_trace_message_bits=20.0,
        noise_estimator_require_trace_safe=False,
        scale_floor_policy=None,
        scale_floor_min_bits=None,
        openevolve_scale_floor_candidates=None,
    ):
        if le_json is None:
            return # should be filled later
        json_parsed = {}
        try:
            with open(le_json, 'r') as file:
                json_parsed = json.load(file)
        except FileNotFoundError:
            raise Exception(f"The file {le_json} was not found.")
        except json.JSONDecodeError:
            raise Exception(f"Error decoding JSON from the file {le_json}.")
        
        self.le_json = le_json
        self.poly_deg = int(json_parsed.get("poly_deg", json_parsed.get("polynomialDegree", 32768)))
        self.max_slot = self.poly_deg // 2
        self.bts_ub = int(json_parsed.get("bootstrapLevelUpperBound", 14))
        self.bts_lb = int(json_parsed.get("bootstrapLevelLowerBound", 1))
        self.lvl_ub = int(json_parsed.get("levelUpperBound", 14))
        self.lvl_lb = int(json_parsed.get("levelLowerBound", 1))
        self.Sf = int(json_parsed.get("rescalingFactor", 51))
        self.backend = json_parsed.get("runtime", "")
        self.latency_table = json_parsed.get("latencyTable", {})
        
        assert mode in ["compile", "execute"], "Mode must be either 'compile' or 'execute'"
        self.mode = mode
        self.sysname = sysname
        self.Sw = Sw if Sw is not None else self.Sf
        self._constant_scale_is_default = CSw is None
        self.Csw = CSw if CSw is not None else self.Sw
        self.bpsdepth = bpsdepth  # possibly None
        self.threads = threads if threads is not None else 16
        self.comp = comp if comp is not None else True
        self.part = part if part is not None else True
        self.reqbp = reqbp if reqbp is not None else False
        self.netname = netname if netname is not None else ""
        self.placement_backend = (
            placement_backend
            if placement_backend is not None
            else json_parsed.get("placement_backend", "openevolve")
        )
        if self.placement_backend not in ("openevolve", "ilp"):
            raise ValueError(
                "placement_backend must be 'openevolve' or 'ilp', "
                f"got {self.placement_backend!r}"
            )
        self.ilp_solver = ilp_solver if ilp_solver is not None else json_parsed.get("ilp_solver", "pulp")
        if self.ilp_solver not in ("gurobi", "pulp"):
            raise ValueError(f"ilp_solver must be 'gurobi' or 'pulp', got {self.ilp_solver!r}")
        if self.placement_backend == "openevolve" and self.ilp_solver == "gurobi":
            # The OpenEvolve path must be usable without importing gurobipy.
            self.ilp_solver = "pulp"
        self.openevolve_config = openevolve_config
        self.openevolve_output_dir = openevolve_output_dir
        self.openevolve_iterations = int(openevolve_iterations or 0)
        self.openevolve_seed = int(openevolve_seed)
        self.openevolve_keep_workdir = bool(openevolve_keep_workdir)
        self.openevolve_provider = (
            openevolve_provider
            if openevolve_provider is not None
            else json_parsed.get("openevolve_provider", "gemini")
        )
        if self.openevolve_provider not in ("gemini", "openai", "custom"):
            raise ValueError(
                "openevolve_provider must be 'gemini', 'openai', or 'custom', "
                f"got {self.openevolve_provider!r}"
            )
        self.openevolve_model = (
            openevolve_model
            if openevolve_model is not None
            else json_parsed.get("openevolve_model", DEFAULT_OPENEVOLVE_GEMINI_MODEL)
        )
        provider_default_api_base = (
            DEFAULT_OPENEVOLVE_GEMINI_API_BASE
            if self.openevolve_provider == "gemini"
            else DEFAULT_OPENEVOLVE_OPENAI_API_BASE
            if self.openevolve_provider == "openai"
            else None
        )
        self.openevolve_api_base = (
            openevolve_api_base
            if openevolve_api_base is not None
            else json_parsed.get("openevolve_api_base", provider_default_api_base)
        )
        self.openevolve_api_key_env = (
            openevolve_api_key_env
            if openevolve_api_key_env is not None
            else json_parsed.get("openevolve_api_key_env", "OPENAI_API_KEY")
        )
        self.openevolve_primary_weight = max(
            0.0,
            float(
                openevolve_primary_weight
                if openevolve_primary_weight is not None
                else json_parsed.get("openevolve_primary_weight", 1.0)
            ),
        )
        self.openevolve_secondary_provider = (
            openevolve_secondary_provider
            if openevolve_secondary_provider is not None
            else json_parsed.get("openevolve_secondary_provider", "none")
        )
        if self.openevolve_secondary_provider not in ("none", "gemini", "openai", "custom"):
            raise ValueError(
                "openevolve_secondary_provider must be 'none', 'gemini', 'openai', or 'custom', "
                f"got {self.openevolve_secondary_provider!r}"
            )
        secondary_default_api_base = (
            DEFAULT_OPENEVOLVE_GEMINI_API_BASE
            if self.openevolve_secondary_provider == "gemini"
            else DEFAULT_OPENEVOLVE_OPENAI_API_BASE
            if self.openevolve_secondary_provider == "openai"
            else None
        )
        self.openevolve_secondary_model = (
            openevolve_secondary_model
            if openevolve_secondary_model is not None
            else json_parsed.get("openevolve_secondary_model", DEFAULT_OPENEVOLVE_OPENAI_MODEL)
        )
        self.openevolve_secondary_api_base = (
            openevolve_secondary_api_base
            if openevolve_secondary_api_base is not None
            else json_parsed.get("openevolve_secondary_api_base", secondary_default_api_base)
        )
        self.openevolve_secondary_api_key_env = (
            openevolve_secondary_api_key_env
            if openevolve_secondary_api_key_env is not None
            else json_parsed.get("openevolve_secondary_api_key_env", "OPENAI_API_KEY")
        )
        self.openevolve_secondary_weight = max(
            0.0,
            float(
                openevolve_secondary_weight
                if openevolve_secondary_weight is not None
                else json_parsed.get("openevolve_secondary_weight", 0.25)
            ),
        )
        if self.openevolve_secondary_provider == "custom" and not self.openevolve_secondary_api_base:
            raise ValueError(
                "openevolve_secondary_api_base is required when openevolve_secondary_provider is 'custom'"
            )
        self.openevolve_harness = (
            openevolve_harness
            if openevolve_harness is not None
            else json_parsed.get("openevolve_harness", "compile")
        )
        if self.openevolve_harness not in ("compile", "partition"):
            raise ValueError(
                "openevolve_harness must be 'compile' or 'partition', "
                f"got {self.openevolve_harness!r}"
            )
        self.openevolve_search_mode = (
            openevolve_search_mode
            if openevolve_search_mode is not None
            else json_parsed.get("openevolve_search_mode", "bootstrap-mcts")
        )
        if self.openevolve_search_mode not in ("legacy", "beam", "bootstrap-mcts"):
            raise ValueError(
                "openevolve_search_mode must be 'legacy', 'beam', or 'bootstrap-mcts', "
                f"got {self.openevolve_search_mode!r}"
            )
        self.openevolve_granularity = (
            openevolve_granularity
            if openevolve_granularity is not None
            else json_parsed.get("openevolve_granularity", "layer-nonlinear")
        )
        if self.openevolve_granularity not in ("compile", "layer-nonlinear"):
            raise ValueError(
                "openevolve_granularity must be 'compile' or 'layer-nonlinear', "
                f"got {self.openevolve_granularity!r}"
            )
        self.openevolve_leniency = (
            openevolve_leniency
            if openevolve_leniency is not None
            else json_parsed.get("openevolve_leniency", "repair")
        )
        if self.openevolve_leniency not in ("repair", "strict"):
            raise ValueError(
                "openevolve_leniency must be 'repair' or 'strict', "
                f"got {self.openevolve_leniency!r}"
            )
        self.openevolve_max_unit_samples = max(
            1,
            int(
                openevolve_max_unit_samples
                if openevolve_max_unit_samples is not None
                else json_parsed.get("openevolve_max_unit_samples", 64)
            ),
        )
        self.openevolve_eval_suite = (
            openevolve_eval_suite
            if openevolve_eval_suite is not None
            else json_parsed.get("openevolve_eval_suite", "polybert-sampled")
        )
        if self.openevolve_eval_suite not in ("toy", "polybert-sampled", "polybert-full"):
            raise ValueError(
                "openevolve_eval_suite must be 'toy', 'polybert-sampled', or "
                f"'polybert-full', got {self.openevolve_eval_suite!r}"
            )
        self.openevolve_reference_json = (
            openevolve_reference_json
            if openevolve_reference_json is not None
            else json_parsed.get("openevolve_reference_json")
        )
        self.openevolve_finalists = int(
            openevolve_finalists
            if openevolve_finalists is not None
            else json_parsed.get("openevolve_finalists", 3)
        )
        self.openevolve_budget_aggressive = bool(
            openevolve_budget_aggressive
            if openevolve_budget_aggressive is not None
            else json_parsed.get("openevolve_budget_aggressive", True)
        )
        self.openevolve_target_bootstrap_count = max(
            0,
            int(
                openevolve_target_bootstrap_count
                if openevolve_target_bootstrap_count is not None
                else json_parsed.get("openevolve_target_bootstrap_count", 0)
            ),
        )
        self.openevolve_llm_timeout_sec = max(
            1,
            int(
                openevolve_llm_timeout_sec
                if openevolve_llm_timeout_sec is not None
                else json_parsed.get("openevolve_llm_timeout_sec", 180)
            ),
        )
        self.openevolve_llm_retries = max(
            0,
            int(
                openevolve_llm_retries
                if openevolve_llm_retries is not None
                else json_parsed.get("openevolve_llm_retries", 1)
            ),
        )
        self.openevolve_llm_retry_delay_sec = max(
            0,
            int(
                openevolve_llm_retry_delay_sec
                if openevolve_llm_retry_delay_sec is not None
                else json_parsed.get("openevolve_llm_retry_delay_sec", 2)
            ),
        )
        self.openevolve_evaluator_timeout_sec = max(
            1,
            int(
                openevolve_evaluator_timeout_sec
                if openevolve_evaluator_timeout_sec is not None
                else json_parsed.get("openevolve_evaluator_timeout_sec", 180)
            ),
        )
        self.openevolve_parallel_evaluations = max(
            1,
            int(
                openevolve_parallel_evaluations
                if openevolve_parallel_evaluations is not None
                else json_parsed.get("openevolve_parallel_evaluations", 1)
            ),
        )
        self.openevolve_checkpoint_interval = max(
            1,
            int(
                openevolve_checkpoint_interval
                if openevolve_checkpoint_interval is not None
                else json_parsed.get("openevolve_checkpoint_interval", 5)
            ),
        )
        self.openevolve_llm_max_tokens = max(
            256,
            min(
                50_000,
                int(
                    openevolve_llm_max_tokens
                    if openevolve_llm_max_tokens is not None
                    else json_parsed.get(
                        "openevolve_llm_max_tokens",
                        DEFAULT_OPENEVOLVE_LLM_MAX_TOKENS,
                    )
                ),
            ),
        )
        self.openevolve_fail_open = bool(
            openevolve_fail_open
            if openevolve_fail_open is not None
            else json_parsed.get("openevolve_fail_open", True)
        )
        self.openevolve_reuse_output = bool(
            openevolve_reuse_output
            if openevolve_reuse_output is not None
            else json_parsed.get("openevolve_reuse_output", False)
        )
        self.noise_estimator = (
            noise_estimator
            if noise_estimator is not None
            else json_parsed.get("noise_estimator", "finalists")
        )
        if self.noise_estimator not in ("off", "finalists"):
            raise ValueError(
                "noise_estimator must be 'off' or 'finalists', "
                f"got {self.noise_estimator!r}"
            )
        self.noise_estimator_binary = (
            noise_estimator_binary
            if noise_estimator_binary is not None
            else json_parsed.get("noise_estimator_binary")
        )
        self.noise_estimator_timeout_sec = max(
            1,
            int(
                noise_estimator_timeout_sec
                if noise_estimator_timeout_sec is not None
                else json_parsed.get("noise_estimator_timeout_sec", 30)
            ),
        )
        self.noise_estimator_min_output_margin_bits = float(
            noise_estimator_min_output_margin_bits
            if noise_estimator_min_output_margin_bits is not None
            else json_parsed.get("noise_estimator_min_output_margin_bits", 2.0)
        )
        self.noise_estimator_alpha = float(
            noise_estimator_alpha
            if noise_estimator_alpha is not None
            else json_parsed.get("noise_estimator_alpha", 14.0)
        )
        self.noise_estimator_max_trace_message_bits = float(
            noise_estimator_max_trace_message_bits
            if noise_estimator_max_trace_message_bits is not None
            else json_parsed.get("noise_estimator_max_trace_message_bits", 20.0)
        )
        self.noise_estimator_require_trace_safe = bool(
            noise_estimator_require_trace_safe
            if noise_estimator_require_trace_safe is not None
            else json_parsed.get("noise_estimator_require_trace_safe", False)
        )
        self.openevolve_compile_hints = None
        self.openevolve_evaluating_candidate = False

        self.resilience_profile_path = resilience_profile
        self.resilience_profile = ResilienceProfile.load(resilience_profile)
        assert resilience_constraint_policy in ["relax-only", "hard-tau"], (
            "resilience_constraint_policy must be either 'relax-only' or 'hard-tau'"
        )
        self.resilience_constraint_policy = resilience_constraint_policy
        if (
            self.resilience_constraint_policy == "relax-only"
            and self.resilience_profile is not None
        ):
            relaxed_sw = self.resilience_profile.relaxed_global_scale(self.Sw)
            if relaxed_sw < self.Sw:
                print(
                    "Resilience relax-only policy lowered guided waterline "
                    f"from Sw={self.Sw} to Sw={relaxed_sw}."
                )
                self.Sw = relaxed_sw
                if self._constant_scale_is_default:
                    self.Csw = relaxed_sw
        self.scale_floor_policy = (
            scale_floor_policy
            if scale_floor_policy is not None
            else json_parsed.get("scale_floor_policy", "waterline")
        )
        if self.scale_floor_policy not in ("waterline", "estimator-relaxed"):
            raise ValueError(
                "scale_floor_policy must be 'waterline' or 'estimator-relaxed', "
                f"got {self.scale_floor_policy!r}"
            )
        default_floor = (
            max(24, int(self.Sw) - 12)
            if self.scale_floor_policy == "estimator-relaxed"
            else int(self.Sw)
        )
        self.scale_floor_min_bits = max(
            0,
            min(
                int(self.Sw),
                int(
                    scale_floor_min_bits
                    if scale_floor_min_bits is not None
                    else json_parsed.get("scale_floor_min_bits", default_floor)
                ),
            ),
        )
        raw_floor_candidates = (
            openevolve_scale_floor_candidates
            if openevolve_scale_floor_candidates is not None
            else json_parsed.get("openevolve_scale_floor_candidates")
        )
        parsed_floor_candidates = _parse_scale_floor_candidates(raw_floor_candidates)
        if self.scale_floor_policy == "estimator-relaxed":
            if parsed_floor_candidates is None:
                parsed_floor_candidates = [
                    int(self.Sw),
                    max(self.scale_floor_min_bits, int(self.Sw) - 4),
                    max(self.scale_floor_min_bits, int(self.Sw) - 8),
                    self.scale_floor_min_bits,
                ]
            candidates = []
            for item in parsed_floor_candidates:
                value = max(self.scale_floor_min_bits, min(int(self.Sw), int(item)))
                if value not in candidates:
                    candidates.append(value)
            if int(self.Sw) not in candidates:
                candidates.insert(0, int(self.Sw))
            self.openevolve_scale_floor_candidates = candidates or [int(self.Sw)]
        else:
            self.openevolve_scale_floor_candidates = [int(self.Sw)]
        self._active_scale_floor_bits = None
        assert resilience_mode in ["waterline", "error-state"], (
            "resilience_mode must be either 'waterline' or 'error-state'"
        )
        self.resilience_mode = resilience_mode
        self.allow_empty_resilience_match = allow_empty_resilience_match
        assert resilience_decomposition in ["off", "bounded-dp"], (
            "resilience_decomposition must be either 'off' or 'bounded-dp'"
        )
        self.resilience_decomposition = resilience_decomposition
        self.resilience_decompose_threshold = resilience_decompose_threshold
        self.resilience_max_boundary_states = resilience_max_boundary_states
        self.resilience_error_buckets = resilience_error_buckets
        self.ilp_task_time_limit_sec = ilp_task_time_limit_sec
        self.resilience_match_report = None
        self.resilience_decomposition_report = None
        self.resilience_error_model = DEFAULT_RESILIENCE_ERROR_MODEL.copy()
        self.resilience_error_model.update(json_parsed.get("resilienceErrorModel", {}))
        
        self.trunc_val = 1 # truncation value for latency estimation
        self.dacapo_mlir_in = True  # need to revert the input MLIR level
        self.dacapo_mlir_out = True # need to revert the output MLIR level

    def has_resilience_constraints(self) -> bool:
        return (
            self.resilience_profile is not None
            and len(self.resilience_profile.constraints) > 0
        )

    def use_bounded_resilience_decomposition(self) -> bool:
        return (
            self.resilience_mode == "error-state"
            and self.resilience_decomposition == "bounded-dp"
            and self.has_resilience_constraints()
            and self.part
        )

    def max_scale(self) -> int:
        return int(self.Sf + 2 * max(self.Sw, self.Csw))

    def decryptable_scale_bound(self, level: int) -> int:
        """Maximum scale bits decryptable at ``level`` under Orbit's ILP model."""
        return int(self.Sf * (int(level) - self.lvl_lb + 2) - 7)

    def boundary_output_scale_bound(self, level: int) -> int:
        """Lattigo output boundary bound used by Orbit's IO-budget ILP constraint."""
        return int(self.Sf * (int(level) + 1) - 7)

    def is_decryptable_state(self, level: int, scale: int) -> bool:
        level = int(level)
        scale = int(scale)
        return (
            self.lvl_lb <= level <= self.lvl_ub
            and 0 <= scale <= self.max_scale()
            and scale <= self.decryptable_scale_bound(level)
        )

    def active_scale_floor_bits(self) -> int:
        if self.scale_floor_policy != "estimator-relaxed":
            return int(self.Sw)
        raw_floor = getattr(self, "_active_scale_floor_bits", None)
        if raw_floor is None:
            return int(self.Sw)
        return max(self.scale_floor_min_bits, min(int(self.Sw), int(raw_floor)))

    def scale_lower_bound(self, node_label: str, node_attrs: dict, port: str) -> int:
        active_floor = self.active_scale_floor_bits()
        if self.resilience_profile is None:
            return active_floor
        profile_bound = self.resilience_profile.scale_lower_bound(
            node_label,
            node_attrs,
            self.Sw,
            port,
        )
        if self.resilience_constraint_policy == "hard-tau":
            return profile_bound
        if self.resilience_constraint_policy == "relax-only":
            return min(active_floor, profile_bound)
        return profile_bound
        
    def check_res(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        if in_lvl < out_lvl:
            return False
        if self.Sf * in_lvl - in_scl < self.Sf * out_lvl - out_scl:
            return False
        return True
    
    def check_resbts(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        if self.check_res(in_lvl, in_scl, out_lvl, out_scl):
            return True
        if not self.check_res(in_lvl, in_scl, self.bts_lb, self.Sf):
            return False
        r = max(0, int(math.ceil((self.Sf - out_scl) / max(self.Sf, 1))))
        if not (self.bts_lb < out_lvl + r <= self.bts_ub):
            return False
        if not self.check_res(out_lvl+r, self.Sf, out_lvl, out_scl):
            return False
        return True
