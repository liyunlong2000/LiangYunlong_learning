# docker部署neo4j
## 拉取镜像
从镜像源中找到合适的镜像
```
sudo docker search neo4j
```
拉取合适的镜像
```
sudo docker pull neo4j(:版本号)
```
查看本地镜像，检验是否拉取成功
```
sudo docker images
```
## 构建neo4j容器
创建一个目录用于文件挂载，在当前目录下创建以下子目录：
- data——数据存放的文件夹
- logs——运行的日志文件夹
- import——为了大批量导入csv来构建数据库，需要导入的节点文件nodes.csv和关系文件rel.csv需要放到这个文件夹下）
```
sudo docker run -d --name neo4j_test\   //-d表示容器后台运行 --name指定容器名字,为neo4j_test
	-p 7474:7474 -p 7687:7687 \           //映射容器的端口号到宿主机的端口号
	-v /home/neo4j/data:/data \           //把容器内的数据目录挂载到宿主机的对应目录下
	-v /home/neo4j/logs:/logs \           //挂载日志目录
	-v /home/neo4j/import:/var/lib/neo4j/import \  //挂载数据导入目录
	--env NEO4J_AUTH=neo4j/admin \     //设定数据库的名字为neo4j，访问密码为admin
	neo4j //指定使用的镜像
```
下面为可以直接终端执行的命令
```
sudo docker run -d --name neo4j_test -p 7474:7474 -p 7687:7687 -v /home/neo4j/data:/data -v /home/neo4j/conf:/var/lib/neo4j/conf  -v /home/neo4j/import:/var/lib/neo4j/import --env NEO4J_AUTH=neo4j/admin neo4j
```
 ## neo4j配置
 使用以下配置使得浏览器能够访问neo4j数据库。
 ```
// 进入容器配置目录进行以下更改
//在文件配置末尾添加这一行
dbms.connectors.default_listen_address=0.0.0.0  //指定连接器的默认监听ip为0.0.0.0，即允许任何ip连接到数据库

//修改
dbms.connector.bolt.listen_address=0.0.0.0:7687  //取消注释并把对bolt请求的监听“地址:端口”改为“0.0.0.0:7687”
dbms.connector.http.listen_address=0.0.0.0:7474  //取消注释并把对http请求的监听“地址:端口”改为“0.0.0.0:7474”
```
 保存后退出，重启neo4j容器。
```
docker restart 容器id（或者容器名）
```
### 防火墙设置
```
// 查看当前防火墙状态，若为“inactive”，则防火墙已关闭，不必进行接续操作。
sudo ufw status

// 若防火墙状态为“active”，则使用下列命令开放端口
sudo ufw allow 7474
sudo ufw allow 7687

// 重启防火墙
sudo ufw reload
```
# neo4j数据导入
```
// 数据准备
清空data/databases/graph.db文件夹(如果有),将清洗好的结点文件nodes.csv和关系文件rel.csv拷贝到宿主机/home/neo4j/import中

// docker以exec方式进入容器的交互式终端
docker exec -it container_name(or container_id) /bin/bash

// 停掉neo4j
bin/neo4j stop

//使用如下命令导入
bin/neo4j-admin import \
	--database=graph.db \	        //指定导入的数据库，没有系统则会在data/databases下自动创建一个
	--nodes ./import/nodes.csv 		//指定导入的节点文件位置
	--relationships ./import/rel.csv //指定导入的关系文件位置
	--skip-duplicate-nodes=true 	//设置重复节点自动过滤
	--skip-bad-relationships=true 	//设置bad关系自动过滤
	
//可执行一行式终端命令
bin/neo4j-admin import --database=graph.db --nodes ./import/nodes.csv --relationships ./import/rel.csv --skip-duplicate-nodes=true --skip-bad-relationships=true

// 容器内启动neo4j
bin/neo4j start

// 退出交互式终端但是保证neo4j后台继续运行
ctrl + P + Q

//保险起见，重启neo4j容器
docker restart container_name(or container_id)
```
# 问题记录
- 无法挂载配置目录，否则无法启动容器
- 使用VS远程连接时，需要手动设置端口，以便浏览器能够正确访问
- 2022/10/3拉取的neo4j容器不需要再配置`neo4j.conf`
# 参考链接
[docker安装部署neo4j](https://www.cnblogs.com/caoyusang/p/13610408.html)
