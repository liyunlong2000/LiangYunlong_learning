# 环境
-	IntelliJ IDEA Community Edition 2022.2.3
-	Maven 3.6.3
-	OpenJDK version 11.0.2 
# 配置
将项目`neo4j-master`导入`idea`中,点击`maven`中`install`选项,安装`jar`包到本地.

<img src="https://user-images.githubusercontent.com/56336922/195968479-2a516e67-80d6-4f47-a929-11ac2ccf1182.png" width="200" height="400" />

打包完成后的`jar`包在目录`neo4j-master\packaging\standalonetarget`中,并将`.zip`文件解压到当前目录.
# 启动
找到入口类:`org.neo4j.server.CommunityEntryPoint`
-	位于`neo4j-4.4/community/neo4j/src/main/java/org/neo4j/server/CommunityEntryPoint.java`

点击运行，终端报错，提示需要提供参数`--home-dir`.因此在运行选项中加入以下参数.
```
-server --home-dir=D:\code\neo4j\neo4j-master\packaging\standalone\target\neo4j-community-4.4.12-SNAPSHOT --config-dir=D:\code\neo4j\neo4j-master\packaging\standalone\target\neo4j-community-4.4.12-SNAPSHOT\conf
```
-	`--home-dir `：neo4j解压后的根目录
-	`--config-dir `：neo4j解压后的配置目录,在根目录下的`bin`目录中
再次点击运行按钮,neo4j成功启动.

![image](https://user-images.githubusercontent.com/56336922/195968698-49599fef-c8e9-41a5-b05a-ade32c353508.png)
