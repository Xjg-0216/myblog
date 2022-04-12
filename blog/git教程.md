## git 教程

#### 1、创建本地的版本库

![image-20220412183643545](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412183643545.png)

创建demo文件夹

进入到demo文件夹，把该文件夹变成git可管理的仓库（单击右键，选择git bush--->>> git init)

![image-20220412183945509](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412183945509.png)

之后，在该文件夹会多出一个.git文件夹

将文件添加到缓存区

* 通过git add . 可以将该文件夹下所有的文件添加到缓存区
* 通过git add filename， 可以将具体的文件添加到缓存区

![image-20220412184305946](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412184305946.png)

我们将git教程.md文件添加到缓存区

通过git status查看缓存区状态

![image-20220412184420425](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412184420425.png)

为缓存区文件添加注释(git commit -m "context")

![image-20220412184551434](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412184551434.png)

### 2、连接远程仓库(github或者gitee)

[见该教程](https://blog.csdn.net/generallizhong/article/details/94014779?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164974948216780271913967%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164974948216780271913967&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-94014779.142^v7^pc_search_result_cache,157^v4^control&utm_term=git%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6%E5%88%B0github%E4%BB%93%E5%BA%93&spm=1018.2226.3001.4187)

`git remote -v`查看远程库信息：

![image-20220412190410552](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412190410552.png)



### 3、上传内容

git push origin master

![image-20220412190606498](https://gitee.com/xjg0216/blogimg/raw/master/img/image-20220412190606498.png)

这是由于你新创建的那个仓库里面的README文件不在本地仓库目录中，这时我们可以通过以下命令先将内容合并以下：

git pull --rebase origin master

