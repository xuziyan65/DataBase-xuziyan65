create database my_database;
CREATE TABLE product (     
`Material` VARCHAR(50) PRIMARY KEY,             
`Describrition` TEXT,                                    
`面价` DECIMAL(12,2),                           
`折扣` DECIMAL(5,2),                          
`币种` VARCHAR(10),                             
`出厂价_含税` INT,                             
`出厂价_不含税` INT,                            
`单位` VARCHAR(10),                            
`仓位描述` VARCHAR(50),                        
`产品分类` VARCHAR(100),                      
`维护单位` VARCHAR(10),                        
`价格货币` VARCHAR(10),                        
`客户类型` VARCHAR(50),                         
`是否参与返点` VARCHAR(20),                     
`流程编码` VARCHAR(50),                         
`有效期`DATE                          
)
SELECT * FROM my_database.product;