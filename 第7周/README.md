# LDBC SNB BI测试
## 关于LDBC和SNB测试
LDBC（Linked Data Benchmark Council，国际关联数据基准委员会）是图数据库领域权威的基准指南制定者与测试标准发布机构，旨在制定一套公平、诚信、可对比的方法和机制来衡量图数据库管理系统，共同推进这项前沿技术的发展。

SNB（Social Network Benchmark，社交网络基准测试）是LDBC开发的基于社交网络的基准测试之一。其包含两种场景：
- 交互场景(Interactive)：关注交互式事务查询
- 商务智能场景(Business Intelligence)：关注分析查询

## 测试流程
本节主要介绍LDBC SNB BI的测试的过程，需要提前下载配置[ldbc\_snb\_bi](https://github.com/ldbc/ldbc_snb_bi)、[ldbc\_snb\_datagen\_spark](https://github.com/ldbc/ldbc_snb_datagen_spark)。
### 生成数据集
进入数据生成器ldbc\_snb\_datagen\_spark目录，设置环境变量。其中SF为数据规模。
```
export SF=0.1
export LDBC_SNB_DATAGEN_MAX_MEM=50G
export LDBC_SNB_DATAGEN_JAR=$(sbt -batch -error 'print assembly / assemblyOutputPath')
```
使用datagen镜像生成数据集，生成的数据将放在对应的容器out-sf\$\{SF\}目录下.
```
rm -rf out-sf${SF}/
sudo docker run \
ldbc/datagen:latest\
    --memory ${LDBC_SNB_DATAGEN_MAX_MEM} \
    -- \
    --format csv \
    --scale-factor ${SF} \
    --explode-edges \
    --mode bi \
    --output-dir out-sf${SF}/ \
    --generate-factors \
    --format-options header=false,quoteAll=true,compression=gzip
```
将容器内生成的数据集复制到数据生成器ldbc\_snb\_datagen\_spark目录下,数据在out-sf0.1目录下.
```
sudo docker cp 9a6333d2d7f5:out-sf${SF}  /home/lyl/neo4jtest1/ldbc_snb_datagen_spark
```
### 加载数据
使用以下命令，设置数据生成器目录和NEO4J的CSV目录。
```
export LDBC_SNB_DATAGEN_DIR=/home/lyl/neo4jtest1/ldbc_snb_datagen_spark
export NEO4J_CSV_DIR=/home/lyl/neo4jtest1/ldbc_snb_datagen_spark/out-sf0.1/graphs/csv/bi/composite-projected-fk/
```
使用以下命令，配置Neo4j的可用内存。
```
export NEO4J_ENV_VARS="${NEO4J_ENV_VARS-} --env NEO4J_dbms_memory_pagecache_size=20G --env NEO4J_dbms_memory_heap_max__size=20G"
```
进入ldbc\_snb\_bi/cypher目录下，使用脚本加载数据集。数据将加载到容器lyl\_neo4j中，可在var.sh中设置容器的名称。
```
scripts/load-in-one-step.sh
```
### 参数生成
进入ldbc\_snb\_bi/paramgen/目录,使用以下脚本安装所需的依赖。
```
scripts/install-dependencies.sh
```
进入数据生成器ldbc\_snb\_datagen\_spark目录设置以下环境变量.
```
export SF=0.1
export LDBC_SNB_DATAGEN_MAX_MEM=50G
export LDBC_SNB_DATAGEN_JAR=$(sbt -batch -error 'print assembly / assemblyOutputPath')
```
使用datagen生成实体因子.
```
rm -rf out-sf${SF}/
sudo docker run \
ldbc/datagen:latest\
    --memory ${LDBC_SNB_DATAGEN_MAX_MEM} \
    -- \
    --format parquet \
    --scale-factor ${SF} \
    --mode raw \
    --output-dir out-sf${SF} \
    --generate-factors
```
使用以下命令,将容器中生成的实体因子放到本地.
```
sudo docker cp ae8994f5f328:/out-sf${SF} /home/lyl/neo4jtest1/ldbc_snb_datagen_spark
```
在ldbc\_snb\_bi/cypher/paramgen目录下使用以下脚本，将本地的因子复制到ldbc\_snb\_bi/paramgen/factors目录下.
```
scripts/get-factors.sh
```
使用参数生成器脚本，生成查询所需的参数，生成的参数将放置在../para-
meters/parameters-sf\$\{SF\}/目录下
```
scripts/paramgen.sh
```
### 查询
在ldbc\_snb\_bi/cypher/目录下使用以下脚本，执行查询测试。相应的结果将放置在/output/output-sf\$\{SF\}/目录下。
```
scripts/queries.sh ${SF}
```

### 问题记录
- python中pip安装的neo4j包版本为4.3.0，neo4j-driver包版本为1.6.2
- 脚本中docker使用需手动加上sudo
