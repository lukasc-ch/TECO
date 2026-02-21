
## Prompt 1: outlining the plan (claude-code / Opus)
I want to create an agentic code optimizer that can take Triton code and maximize performance for a specific GPU. I later want to extend this to a) take other code like tilelang, AscendC, CUDA, and b) possibly translate that code to a different programming language, c) optimize it for further platforms (e.g., the Huawei Ascend 910B). 

Here is the special capability: I want it to learn from experience. 
It should learn like a very good student: 
1) starting by collecting materials/documentation online about the target language, 
3) go through some tutorials & learn how to use tools like profilers and debuggers, 
2) study the problem at hand and research known approaches, 
4) self-reflect to write-up/summarize its lessons learned. 

Importantly, when a problem is hard and it is difficult to identify directly the limiting factor (e.g., the profiling result is incocnlusive, or there is a lack of documentation of the target language), it should be able to think about how to break down to problem into isolated analysis with the goal to reach a partial conclusion -- so this might mean not just improving the target code, but creating separate kernels to better understand the platform or the task running on there. After this, it should self-reflect about new generalizable conclusions to extend its experience database and specific feedback for the main problem. 







Let's make a plan add some features & make some modifications: 
1) plan to add a mode with much more verbose output, colored in yellow. 
2) if there is a correctness issue, make several debugging attempts
3) make sure to first run the baseline implementation stand-alone to check its correctness
4) have the option to configure two separate models for the strategy planning and the code optimizations phases
5) add performance information to teco itself to better understand how time is spent as it is very slow currently.

