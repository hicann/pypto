### 数据采集前环境变量设置：

L0----------------------------------------------------------------------暂无使用
export PROFILER_SAMPLECONFIG='{"stars_acsq_task":"off","app":"test_dynshape","prof_level":"l0","taskTime":"l0","result_dir":"/home/chenjz/tilefwk_code/build","app_dir":"/home/chenjz/tilefwk_code/build/.","ai_core_profiling":"off","aicpuTrace":"on"}'
L1-----------------------------------------------------------------------泳道图数据打点采集
export PROFILER_SAMPLECONFIG='{"stars_acsq_task":"off","app":"test_dynshape","prof_level":"l1","taskTime":"l1","result_dir":"/home/chenjz/tilefwk_code/build","app_dir":"/home/chenjz/tilefwk_code/build/.","ai_core_profiling":"off","aicpuTrace":"on"}'
L2-----------------------------------------------------------------------打点采集泳道图、PMU数据
export PROFILER_SAMPLECONFIG='{"stars_acsq_task":"off","app":"test_dynshape","prof_level":"l2","taskTime":"l2","result_dir":"/home/chenjz/tilefwk_code/build","app_dir":"/home/chenjz/tilefwk_code/build/.","ai_core_profiling":"off","aicpuTrace":"on"}'

result_dir/app_dir: 采集的数据存放路径，自定义设置



### 工具脚本使用：

python3 tilefwk_prof_data_parser.py -p xxx性能数据的路径  -t  #按照task id绘制泳道图

![image-20250209093802313](http://image.huawei.com/tiny-lts/v1/images/hi3ms/06c190b7818fde8ba3d67deb166fa8b6_356x193.png@900-0-90-f.png)

1、采集的数据集

2、数据集解析并绘制绘制泳道图脚本

3、数据集原始存放路径：以环境变量设置为"result_dir":"/home/chenjz/tilefwk_code/build为例

/home/chenjz/tilefwk_code/build/PROF_000001*/device_x/ */**aicpu.data.0.slice_0**

**该脚本暂未调测 **

python3 tilefwk_pmu_to_csv.py -p ./



### 