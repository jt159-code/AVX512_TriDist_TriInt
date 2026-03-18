# 远程控制服务器

### 上传文件夹

scp -r D:\VS\final\AVX512_TriDist_TriInt_3 juting@101.76.222.120:~/my_project

### 登录服务器

ssh juting@101.76.222.120

### 情况文件夹

rm -rf ~/my_project/*

### 编译

 g++ -mavx512f -O2 -o myprogram main.cpp TriDist.cpp TriInt.cpp

g++ -o myprogram main.cpp TriDist.cpp TriInt.cpp -mavx512f

# GitHub

### 1、连接

git init
git remote add origin https://github.com/jt159-code/AVX512_TriDist_TriInt.git
git remote show origin（当前远程连接的仓库）
git fetch（同步远程仓库的最新信息）

### 2、操作分支

#### 确认分支

git branch
git branch -a（查看所有分支（包括远程分支））

#### 切换分支

git checkout master或者git switch master

#### 创建版本分支

git checkout -b version-avx0（本地）
git checkout --orphan version-avx0（本地）
git checkout --orphan version-avx0（远程）

#### 和远程仓库建立关联

git push -u origin version-avx0

#### 删除远程分支

git push origin --delete version-avx0

### 4、.gitignore

#### 恢复

git restore .gitignore

### 5、基础操作

#### 添加文件（.gitignore会排除临时文件）

git add .
git add AVX512_TriDist_TriInt_0/
git add AVX512_TriDist_TriInt_0/

#### 提交

git commit -m "描述你的修改"

#### 推送到 GitHub

git push 
git push -f origin master（强制推送到远程）
git push origin HEAD:refs/heads/version-avx0（新建远程分支，并推送到远程分支）

#### 查看当前状态

git status

#### 删除所有文件（保留.git目录）

git rm -rf .

# 版本变更

PQP_Tridist———基础的，可在VS上运行的版本

AVX512——AI修改了基础运算的，可在VS上运行

AVX512_TriDist_TriInt_0——版本0，可在VScode上运行的，未进行AVX512准备的原始版本

AVX512_TriDist_TriInt_1——版本1，VScode，结构体分组

AVX512_TriDist_TriInt_2——版本2，VScode，8对/16对三角形，AoS->SoA

AVX512_TriDist_TriInt_3——版本3，VScode，AVX512向量化

AVX512_TriDist_TriInt_4——版本4，Linux，无AVX512

AVX512_TriDist_TriInt_5——版本5，Linux，AVX512

AVX512_TriDist_TriInt_6——版本6，Linux，AVX512，数据对齐/数据预取

AVX512_TriDist_TriInt_7——版本7，Linux，无AVX512，真实数据

【Time per pair: 0.283 us】

AVX512_TriDist_TriInt_8——版本8，Linux，AVX512，真实数据



