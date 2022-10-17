# 第一遍调试
找到CommunityEntryPoint,其中关键的代码为`int status = NeoBootstrapper.start( new CommunityBootstrapper(), args );`,在该行打上断点.
然后进入调试模式,使用F7单步调试,F8跳出.
1.  int status = NeoBootstrapper.start( new CommunityBootstrapper(), args );
    1. new CommunityBootstrapper()
        1. private GraphDatabaseDependencies dependencies = GraphDatabaseDependencies.newDependencies();
            1. Iterable<ExtensionFactory<?>> extensions = getExtensions( Services.loadAll( ExtensionFactory.class ).iterator() );
               - 加载服务的所有可用实现
            2. return new GraphDatabaseDependencies( null, null, null, extensions,urlAccessRules, empty() );
    2. public static int start( Bootstrapper boot, String... argv )
        1. boot.start( args.homeDir(), args.configFile(), args.configOverrides(), args.expandCommands() ); 
            1. databaseManagementService = createNeo( config, dependencies );
                1. protected DatabaseManagementService createNeo( Config config, GraphDatabaseDependencies dependencies )
                    1. DatabaseManagementServiceFactory facadeFactory = new DatabaseManagementServiceFactory( COMMUNITY, CommunityEditionModule::new );
                    2. return facadeFactory.build( config, dependencies );
                        - 返回图数据库实例

## 调试 facadeFactory
1. DatabaseManagementServiceFactory facadeFactory = new DatabaseManagementServiceFactory( COMMUNITY, CommunityEditionModule::new );
    - 初始化相关数据库信息、版本信息
2. return facadeFactory.build( config, dependencies );
    1. globalLife.add( edition.createSystemGraphInitializer( globalModule ) );
    2. globalLife.add( new DefaultDatabaseInitializer( databaseManager ) );
    3. globalLife.add( globalModule.getGlobalExtensions() );
    4. globalLife.add( createBoltServer( globalModule, edition, boltGraphDatabaseManagementServiceSPI, databaseManager.databaseIdRepository() ) );
    5. globalLife.add( webServer );
    6. globalLife.add( globalModule.getCapabilitiesService() );
    7. startDatabaseServer( globalModule, globalLife, internalLog, databaseManager, managementService );
        1. databaseManager.initialiseSystemDatabase();
        2. globalLife.start();
            1. init();
            2. for ( LifecycleInstance instance : instances ) {instance.start();}
            
  ## Lifecycle接口
  ```
  /**
 * Lifecycle interface for kernel components. Init is called first,
 * followed by start,
 * and then any number of stop-start sequences,
 * and finally stop and shutdown.
 *
 * As a stop-start cycle could be due to change of configuration, please perform anything that depends on config
 * in start().
 *
 * Implementations can throw any exception. Caller must handle this properly.
 *
 * The primary purpose of init in a component is to set up structure: instantiate dependent objects,
 * register handlers/listeners, etc.
 * Only in start should the component actually do anything with this structure.
 * Stop reverses whatever was done in start, and shutdown finally clears any set-up structure, if necessary.
 */
public interface Lifecycle
{
    void init() throws Exception;

    void start() throws Exception;

    void stop() throws Exception;

    void shutdown() throws Exception;

}
  ```
  `Lifecycle`为内核组件的统一接口,对于每个内核组件.首先使用`init()`方法初始化所依赖的对象、注册资源/监听器.然后调用`start()`方法运行相关组件.
  
  上面的`globalLife`是一个`LifeSupport`,其继承了`Lifecycle`接口,它有一个私有成员,维护着`LifecycleInstance`集合.
  ```
  private volatile List<LifecycleInstance> instances = new ArrayList<>();
  ```
  `LifeSupport`的`start()`、`init()`、`stop()`、`shutdown()`方法相当于一个`for each`循环,循环使用`instances`中的相应方法.
  
  
  
