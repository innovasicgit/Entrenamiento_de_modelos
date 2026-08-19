#Proyecto Tesis Fernando Gutierrez P.
import os
import psutil
import time
import datetime


def cpu_info(  ):
    cpu = psutil.Process( os.getpid() ).cpu_percent ( 1 ) # tasa de uso de CPU en un segundo, unidad
    cpu_per = '% .2f %%'% cpu # se convierte en un porcentaje, mantenga dos decimales
    return ( cpu_per , cpu )


def mem_info(  ):
    mem = psutil.Process( os.getpid() ).memory_percent()
    
    mem_per = '%.2f%%' % mem
    return ( mem_per , mem )


def disk_info() :
    c_info = psutil.disk_usage( '/' )
    c_per = '%.2f%%' % c_info[ 3 ]
    return c_per


def medida_tiempo() :
    tiempo = datetime.datetime.now()
    return tiempo


def tiempo_total( t1 , t2 ) :
    t_total = t2 - t1
    return t_total


def children_process( cpu , mem ) :
    c_list= []
    children_list = psutil.Process( os.getpid() ).children()
    if len( children_list ) > 0:
        for i in range( len ( children_list ) ):
            c_list += [children_list[ i ].pid]
        info_children = add_children( c_list , cpu , mem )
        return info_children
    else:
        return 'Not children'
    


def add_children( c_list , cpu , mem ) :
    cpu_su = cpu[1]
    mem_su = mem[1]
    for i in range( len ( c_list ) ):
        pid = c_list[i]
        cpu_child = psutil.Process( pid ).cpu_percent ( 1 )
        mem_child = psutil.Process( pid ).memory_percent()
#         print( cpu_child , mem_child )
        cpu_su += cpu_child
        mem_su += mem_child
    cpu_total = '%.2f%%' % cpu_su
    mem_total = '%.2f%%' % mem_su
    return ( cpu_total , mem_total )   

def print_info( cpu , mem , disco , t , otro ) :
    if otro == 'Not children':
        var_prin = f'cpu: {cpu[0]}, memoria: {mem[0]}, disco: {disco}, tiempo: {t}'
    if otro != 'Not children':
        var_prin = f'cpu: {otro[0]}, memoria: {otro[1]}, disco: {disco}, tiempo: {t}'
    return var_prin

# if __name__ == '__main__' :
#     t1 = medida_tiempo()
#     
#     cpu = cpu_info()
#     mem = mem_info()
#     disco = disk_info()
#     otro = children_process( cpu , mem )
#     t2 = medida_tiempo()
#     t= tiempo_total( t1, t2 )
#     print_info( cpu , mem , disco , t , otro )
    
    
    
