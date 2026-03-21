PQP_Tridist———基础的，可在VS上运行的版本

AVX512——AI修改了基础运算的，可在VS上运行

AVX512_TriDist_TriInt_0——版本0，可在VScode上运行的，未进行AVX512准备的原始版本

AVX512_TriDist_TriInt_1——版本1，VScode，结构体分组

AVX512_TriDist_TriInt_2——版本2，VScode，8对/16对三角形，AoS->SoA

AVX512_TriDist_TriInt_3——版本3，VScode，AVX512向量化

AVX512_TriDist_TriInt_4——版本4，Linux，无AVX512

【Average time per triangle pair: 0.000700 ms】

AVX512_TriDist_TriInt_5——版本5，Linux，AVX512

AVX512_TriDist_TriInt_6——版本6，Linux，AVX512，数据对齐/数据预取

【Average time per triangle pair: 0.000066 ms】

AVX512_TriDist_TriInt_7——版本7，Linux，AVX512，真实数据

【默认： Time per pair: 0.380332 us】

【O1：Time per pair: 0.043375 us】

【O2：Time per pair: 0.039795 us】

【O3： Time per pair: 0.034886 us】

AVX512_TriDist_TriInt_8——版本8，Linux，无AVX512，真实数据（TriDist.cpp版本上传错了，直接使用版本4的TriDist.cpp即可）

【O0：Time per pair: 0.868 us】

【O1：Time per pair: 0.269 us】

【O2：Time per pair: 0.335 us】

【O3：Time per pair: 0.308 us】
