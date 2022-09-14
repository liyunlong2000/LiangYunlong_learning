# docker案例
## 部署web项目
### 拉取tomcat镜像
```
docker pull tomcat
```
使用上述命令拉取最新版的tomcat镜像
```
docker images
```
使用上述命令查看拉取到的tomcat镜像
![image](https://user-images.githubusercontent.com/56336922/190150161-72cab68d-6550-422b-acaf-b3a414e80e55.png)

### 生成容器
```
docker run --name tomcat -p 8080:8080 -d tomcat
```
使用上述命令生成容器
- --name：生成容器的名称为tomcat
- -p：将容器8080端口映射为服务器的8080端口
- -d：容器在后台运行
- tomcat：镜像的名称

```
docker ps
```
使用上述命令查看生成的容器
![image](https://user-images.githubusercontent.com/56336922/190150705-bfa3feee-738d-4710-85a6-0c683dcabe81.png)
### 访问tomcat
浏览器中输入URL:`ip:8080`访问tomcat
![image](https://user-images.githubusercontent.com/56336922/190150810-41e89c82-1ef2-4396-bcf6-ff74a2b55953.png)
因为拉取的tomcat中webapps文件夹是空的，所以访问不到默认资源。需要进入到容器中，将文件夹`webapps`删除，并将`webapps.dist`文件夹改成
`webapps`
```
rm -rf webapps/
mv webapps.dist/ webapps/
```
完成上述操作后，在浏览器中再次输入URL:`ip:8080`访问tomcat
![image](https://user-images.githubusercontent.com/56336922/190156455-16f40d0c-18ec-4920-8fb0-dcdf67f43a1d.png)
