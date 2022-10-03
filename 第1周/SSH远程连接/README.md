# SSH连接远程服务器
1. 按照服务器配置指南进行操作
2. 相应配置文件为：
```
Host lab
    HostName 222.195.92.204
    User lyl
    Port 2424
    IdentityFile ~/.ssh/id_rsa
```
- lab：自定义的主机，设置后可使用命令`ssh lab`连接
- HostName：远程服务器的IP、域名
- User：自己的用户名
- Port：远程服务器SSH监听的端口(默认为22)
- IdentityFile：认证文件的位置，可使用上述写法，需要将文件id_rsa所在的目录加入环境变量Path中
3. 按照文档方法远程连接到服务器
