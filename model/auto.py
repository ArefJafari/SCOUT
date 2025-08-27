from transformers import AutoConfig, AutoModelForCausalLM
from .scout_swa.configuration_scout_swa import ScoutSWAConfig
from .scout_swa.modeling_scout_swa import ScoutSWAForCausalLM
from .scout_mamba.configuration_scout_mamba import ScoutMambaConfig
from .scout_mamba.modeling_scout_mamba import ScoutMambaForCausalLM

# Explicit registration for local models
def register_models():
    AutoConfig.register("scout_swa", ScoutSWAConfig)
    AutoModelForCausalLM.register(ScoutSWAConfig, ScoutSWAForCausalLM)
    AutoConfig.register("scout_mamba", ScoutMambaConfig)
    AutoModelForCausalLM.register(ScoutMambaConfig, ScoutMambaForCausalLM)



