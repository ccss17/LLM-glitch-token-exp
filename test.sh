#!/bin/bash

# set -xeuo
set -x

# pixi run extract_exaone_3_0 
# # pixi run extract_exaone_4_0_1_32b 
# # pixi run extract_hyperclovax 
# pixi run extract_hyperclovax_think 
# # pixi run extract_kullm3 
# # pixi run extract_kanana 
# # pixi run extract_solar 
pixi run extract_kanana_a3b
pixi run extract_kanana_safeguard_8b
pixi run extract_exaone_3_5_32b
pixi run extract_solar_instruct_v1_0

# pixi run inject_exaone_3_0 
# # pixi run inject_exaone_4_0_1_32b 
# # pixi run inject_hyperclovax 
# pixi run inject_hyperclovax_think 
# # pixi run inject_kullm3 
# # pixi run inject_kanana 
# # pixi run inject_solar
pixi run inject_kanana_a3b
pixi run inject_kanana_safeguard_8b
pixi run inject_exaone_3_5_32b
pixi run inject_solar_instruct_v1_0

# pixi run extract_exaone_3_0 --overwrite True
# pixi run extract_exaone_4_0_1_32b --overwrite True
# # pixi run extract_hyperclovax --overwrite True
# pixi run extract_hyperclovax_think --overwrite True
# pixi run extract_kullm3 --overwrite True
# pixi run extract_kanana --overwrite True
# pixi run extract_solar --overwrite True

# pixi run glitch_tokens_exaone_3_0_test 
# pixi run glitch_tokens_exaone_4_0_1_32b_test 
# pixi run glitch_tokens_hyperclovax_test 
# pixi run glitch_tokens_hyperclovax_think_test 
# pixi run glitch_tokens_kullm3_test 
# pixi run glitch_tokens_kanana_test 
# pixi run glitch_tokens_solar_test

# pixi run inject_hyperclovax_test 
# pixi run inject_exaone_3_0_test 
# pixi run inject_exaone_4_0_1_32b_test 
# pixi run inject_hyperclovax_think_test 
# pixi run inject_kullm3_test 
# pixi run inject_kanana_test 
# pixi run inject_solar_test

# pixi run glitch_tokens_exaone_3_0 
# # pixi run glitch_tokens_exaone_4_0_1_32b 
# # pixi run glitch_tokens_hyperclovax  # ERROR
# pixi run glitch_tokens_hyperclovax_think 
# # pixi run glitch_tokens_kullm3 
# # pixi run glitch_tokens_kanana 
# # pixi run glitch_tokens_solar
pixi run glitch_tokens_kanana_a3b
pixi run glitch_tokens_kanana_safeguard_8b
pixi run glitch_tokens_exaone_3_5_32b
pixi run glitch_tokens_solar_instruct_v1_0