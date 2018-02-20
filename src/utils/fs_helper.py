"""
Created on Jun 20, 2017

@author: hnguyen6
"""
import os
import shutil
import subprocess
import glob
import uuid
import Queue
import threading
import operator

s3_protocol = 's3n://'
hdfs_protocol = 'hdfs://'
shared_queue = Queue.Queue()


def empty_shared_queue():
    """
    get all item from the shared queue
    """
    while not shared_queue.empty():
        try:
            shared_queue.get(block=True, timeout=0.1)
        except Queue.Empty:
            break
        shared_queue.task_done()


def local_extract_path(full_path=''):
    """
    Extract full path by removing elements in front of `://` if input generic path is S3 or HDFS
    :param full_path: generic path
    """
    if full_path.startswith(s3_protocol) or full_path.startswith(hdfs_protocol):
        destination_local_path = full_path.split('://', 1)[1]
    else:
        destination_local_path = full_path
    return destination_local_path


def local_split_path(local_path):
    """
    Split local path into list of elements. Each element consists of `(parent path, folder name)`
    :param local_path: local path
    """
    # print '\n[****] split path', local_path
    folder_elements = []
    drive, parent_path = os.path.splitdrive(local_path)
    while True:
        old_path = parent_path
        parent_path, folder_name = os.path.split(parent_path)
        if parent_path != '' and parent_path != old_path:
            folder_elements.insert(0, (drive + parent_path, folder_name))
        else:
            break
    return folder_elements


def local_copy_file(src_file, dst_file):
    """
    Copy local file from source to destination
    :param src_file: source file. Can be a folder
    :param dst_file: destination. Can be a file
    """
    for file_name in glob.glob(src_file):
        shutil.copy(file_name, dst_file)


def local_copy_tree(src_folder, dst_folder):
    """
    Copy a tree folder
    :param src_folder: source folder
    :param dst_folder: destination folder to be parent folder
    """
    dst_folder_child = os.path.join(dst_folder, os.path.basename(src_folder))
    local_remove_folder(dst_folder_child)
    shutil.copytree(src_folder, dst_folder_child)


def local_make_folder(full_path):
    """
    Create all folder elements in full path
    :param full_path: input path
    """
    # print '\n[****] make folder', full_path
    if not os.path.exists(full_path):
        os.makedirs(full_path)


def local_make_folder_fresh(full_path):
    """
    Create folder elements in a full path. If folder exists, delete its content
    :param full_path: input path
    """
    # print '\n[****] build folder path', full_path
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    else:
        local_make_folder(full_path)


def local_path_with_protocol(full_path=''):
    """
    Add `file://` to input path if not S3 or HDFS
    :param full_path: input path
    """
    if not full_path.startswith(s3_protocol) and not full_path.startswith(hdfs_protocol):
        abs_path = os.path.abspath(full_path)
        posix_path = fs_convert_path_to_posix(abs_path).replace('//', '/')
        destination_path = 'file://' + posix_path
        if not destination_path.startswith('file:///'):
            destination_path = destination_path.replace('file://', 'file:///')
            print '[----] path with protocol', destination_path
    else:
        destination_path = fs_normalize_s3_path(full_path)
    return destination_path


def local_remove_file_pattern(full_path, name_pattern):
    """
    Remove files in folder not recursively
    :param full_path: folder path
    :param name_pattern: name pattern of files to be removed
    """
    if os.path.exists(full_path):
        file_name_delete = glob.glob(os.path.join(full_path, name_pattern))
        for fn in file_name_delete:
            os.remove(fn)


def local_remove_folder(full_path):
    """
    Delete a local folder with all of its content
    :param full_path: folder path
    """
    # print '\n[****] clean folder', full_path
    if full_path is not None and os.path.exists(full_path):
        shutil.rmtree(full_path)


def local_retrieve_files(full_path, only_file=True):
    """
    List all files in a folder recursively
    :param full_path: folder path
    :param only_file: if False then include folders in output
    :return: a list of files
    """
    file_path = []
    for folder_parent, folder_names, file_names in os.walk(full_path):
        for fn in file_names:
            file_path.append(os.path.join(folder_parent, fn))
        if not only_file:
            for dn in folder_names:
                file_path.append(os.path.join(folder_parent, dn))
    if not file_path:
        return None
    return file_path


def hdfs_get_root():
    """
    :return: host name/ip
    """
    ip_host = os.getenv('HOSTNAME', 'local[*]').replace('ip-', '').replace('-', '.')
    hdfs_root = hdfs_protocol + ip_host
    return hdfs_root


def run_system_cmd(args_list):
    """
    Run system command using `subprocess`
    :param args_list: list of command arguments, including command name at the beginning
    :return: output, error message, exit code
    """
    print('\n[>>>>] running system command: {0}'.format(' '.join(args_list)))
    prc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = prc.communicate()
    print 'Exit code', prc.returncode
    if prc.returncode:
        print('\n:::::: return code: %d, error: %s' % (prc.returncode, errors))
        # raise RuntimeError('Error running command: %s. Return code: %d, Error: %s' % (
        #     ' '.join(args_list), prc.returncode, errors))
    return output, errors, prc.returncode


def hdfs_copy_file(src_file, dst_file):
    """
    Copy file on HDFS
    :param src_file: source file
    :param dst_file: destination file
    :return: exit code
    """
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-cp', '-f', '-p', src_file, dst_file])
    return ret


def hdfs_copy_file_to_local(src_file, dst_file):
    """
    Copy file from HDFS to local
    :param src_file: source file on HDFS
    :param dst_file: local path
    :return: exit code
    """
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-copyToLocal', src_file, dst_file])
    return ret


def hdfs_list_folder(hdfs_folder, recursively=False):
    """
    List content of HDFS folder
    :param hdfs_folder: HDFS folder\
    :param recursively: list subfolders
    :return: exit code
    """
    ls_cmd = ['hadoop', 'fs', '-ls']
    if recursively:
        ls_cmd.append('-R')
    ls_cmd.append(hdfs_folder)
    output, _, ret = run_system_cmd(ls_cmd)
    lines = output.split('\n')
    res = []
    for line in lines:
        infos = line.split()
        if len(infos) == 8 and infos[2] == 'hadoop':
            hdfs_item = {'userGroup': infos[2], 'user': infos[3], 'size': float(infos[4]),
                         'date': infos[5], 'time': infos[6], 'path': infos[7],
                         'access': infos[0]}
            res.append(hdfs_item)
    res.sort(key=operator.itemgetter('date', 'time'))
    return res, ret


def hdfs_retrieve_files(hdfs_folder, only_file=True):
    """
    List all files in an HDFS folder recursively
    :param hdfs_folder: HDFS folder
    :param only_file: if False then include folders in output
    :return: file list
    """
    file_path = []
    res, ret = hdfs_list_folder(hdfs_folder, recursively=True)
    if not ret:
        for p in res:
            if not only_file or p['access'].startswith('-'):
                file_path.append(p['path'])
    if not file_path:
        return None
    return file_path


def hdfs_make_folder_fresh(hdfs_folder):
    """
    Create new folder on HDFS. If the folder is existing, clean its content
    :param hdfs_folder: path for HDFS folder
    :return: exit code
    """
    ret = hdfs_make_folder(hdfs_folder)
    if not ret:
        hdfs_remove_folder(os.path.join(hdfs_folder, '*'))
    return ret


def hdfs_make_folder(hdfs_folder):
    """
    Create HDFS folder
    :param hdfs_folder: path to folder to create. Parent folders are created if needed be
    :return: exit code
    """
    ret = hdfs_test_folder_existence(hdfs_folder)
    if ret == 1:
        _, _, ret = run_system_cmd(['hadoop', 'fs', '-mkdir', '-p', hdfs_folder])
    return ret


def hdfs_move_file(src_path, dst_path):
    """
    Move file on HDFS
    :param src_path: source path
    :param dst_path: destination path
    :return: exit code
    """
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-mv', os.path.abspath(src_path), os.path.abspath(dst_path)])
    return ret


def hdfs_move_folder_from_local(data_folder, hdfs_folder):
    """
    Copy folder from local to HDFS
    :param data_folder: local folder
    :param hdfs_folder: HDFS folder
    :return: exit code
    """
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-put', os.path.abspath(data_folder), hdfs_folder])
    return ret


def hdfs_move_folder_to_local(hdfs_folder, data_folder):
    """
    Copy folder from HDFS to local
    :param hdfs_folder: HDFS folder
    :param data_folder: local path
    :return: exit code
    """
    base_name = os.path.join(data_folder, os.path.basename(hdfs_folder))
    if os.path.exists(base_name):
        local_remove_folder(base_name)
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-get', hdfs_folder, os.path.abspath(data_folder)])
    return ret


def hdfs_distributed_copy(hdfs_src, hdfs_dst):
    """
    Inter/intra-cluster copying. Work for HDFS <-> S3
    :param hdfs_src: source folder with protocol
    :param hdfs_dst: destination folder with protocol
    :return: exicode
    """
    ret = hdfs_test_folder_existence(hdfs_src)
    if ret == 0:
        _, _, ret = run_system_cmd(['hadoop', 'distcp', '-overwrite', hdfs_src, hdfs_dst])


def hdfs_remove_folder(hdfs_folder):
    """
    Remove folder on HDFS. Can use to remove file also. Works for HDFS and S3
    :param hdfs_folder: HDFS folder
    :return: exit code
    """
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-rm', '-r', hdfs_folder])
    return ret


def hdfs_test_folder_existence(hdfs_folder):
    """
    Check if folder exists on HDFS
    :param hdfs_folder: HDFS path
    :return: exit code
    """
    _, _, ret = run_system_cmd(['hadoop', 'fs', '-test', '-e', hdfs_folder])
    return ret


def thread_loop_hdfs_remove_file():
    """
    Thread execution function to delete HDFS files
    """
    while not shared_queue.empty():
        try:
            hdfs_file = shared_queue.get(block=True, timeout=0.1)
        except Queue.Empty:
            break
        if hdfs_file is not None:
            hdfs_remove_folder(hdfs_file)
        shared_queue.task_done()


def hdfs_remove_multiple_files(hdfs_file_lists):
    """
    A multi-threading wrapper to remove multiple files in HDFS partition
    :param hdfs_file_lists: a list of paths to hdfs files
    """
    if hdfs_file_lists is None:
        return
    num_thread = min(5, len(hdfs_file_lists))
    if num_thread < 1:
        return
    print '[INFO] removing multiple HDFS files', len(hdfs_file_lists)
    if num_thread < 2:
        hdfs_remove_folder(hdfs_file_lists[0])
    else:
        try:
            empty_shared_queue()
            for hdfs_file in hdfs_file_lists:
                shared_queue.put(hdfs_file)
            threads = []
            for _ in range(num_thread):
                t = threading.Thread(target=thread_loop_hdfs_remove_file)
                threads.append(t)
                t.start()
            shared_queue.join()
        except Exception as e:
            print '[ERRR] failed running multiple threads to delete HDFS files', type(e), e


def hdfs_remove_spark_logs(file_to_keep, keep_inprogress=True):
    """
    Remove Spark logs in HDFS partition at /var/log/spark/apps. If number of log files is large, delete 1000 oldest.
    :param file_to_keep: number of most recent logs to keep
    :param keep_inprogress: if set True, do not delete inprogress logs. Number of logs kept may larger than input
    :return: number of logs deleted
    """
    spark_log_path = '/var/log/spark/apps/'
    lines, ret = hdfs_list_folder(spark_log_path)
    if ret or lines is None:
        return
    print len(lines)
    if len(lines) < file_to_keep:
        return
    spark_log_list = []
    for line in lines:
        if 'path' in line:
            spark_log_list.append(line['path'])
    print '[INFO] Spark log found vs. kept in HDFS', len(spark_log_list), file_to_keep
    if len(spark_log_list) <= file_to_keep:
        return
    print '[INFO] deleting Spark event log, keep_inprogress', keep_inprogress
    first_to_keep = min(1000, len(spark_log_list) - file_to_keep)
    spark_log_to_delete = []
    for spark_log in spark_log_list[0:first_to_keep]:
        if not keep_inprogress or not spark_log.endswith('.inprogress'):
            spark_log_to_delete.append(spark_log)
    hdfs_remove_multiple_files(spark_log_to_delete)


def fs_retrieve_files(full_path, only_file=True):
    """
    Retrieve all files recursively
    :param full_path: input path
    :param only_file: if False then include folders in output
    :return: list of files
    """
    if full_path.startswith(hdfs_protocol):
        return hdfs_retrieve_files(full_path, only_file)
    elif full_path.startswith(s3_protocol):
        print '[ERRR] fs_retrieve_files: not supported file system'
        return None
    else:
        return local_retrieve_files(full_path, only_file)


def fs_add_protocol(full_path, file_system, s3_bucket=None):
    """
    Add protocol representing filesystem to path
    :param full_path: full path
    :param file_system: file system can be `HDFS`, `S3` or other
    :param s3_bucket: S3 bucket used if file system is S3
    :return: new path in the form protocol://path
    """
    if file_system == 'HDFS':
        ip_host = os.getenv('HOSTNAME', 'local[*]').replace('ip-', '').replace('-', '.')
        root = hdfs_protocol + ip_host
    elif file_system == 'S3':
        root = s3_protocol
        if s3_bucket is not None:
            root = root + s3_bucket + '/'
    else:
        root = ''
    return root + os.path.normpath(full_path)


def fs_calculate_data_folder_path(data_folder_parent_pref, data_folder_name, date_stamp):
    """
    Calculate data folder using datestamp
    :param data_folder_parent_pref: parent folder
    :param data_folder_name: data folder prefix
    :param date_stamp: datestamp
    :return: new data folder as `data_folder_parent_pref/data_folder_name_date_stamp`
    """
    data_folder_parent = data_folder_parent_pref + '_' + date_stamp
    data_folder = os.path.join(data_folder_parent, data_folder_name)
    return data_folder


def fs_copy_file(src_file, dst_file):
    """
    Copy file within filesystem. Work for local filesystem and HDFS
    :param src_file: source file
    :param dst_file: destination file
    """
    if src_file.startswith(hdfs_protocol):
        hdfs_copy_file(src_file, dst_file)
    elif src_file.startswith(s3_protocol):
        print '[ERRR] fs_copy_file: not supported file system'
    else:
        local_copy_file(src_file, dst_file)


def fs_copy_file_from_local(src_file, dst_file):
    """
    Copy file with source is local folder. Work for local and HDFS
    :param src_file: file in local filesystem
    :param dst_file: destination file, can be local or HDFS
    """
    if dst_file.startswith(hdfs_protocol):
        hdfs_move_folder_from_local(src_file, dst_file)
    elif dst_file.startswith(s3_protocol):
        print '[ERRR] fs_copy_file_from_local: not supported file system'
    else:
        local_copy_tree(src_file, dst_file)


def fs_copy_file_to_local(src_file, dst_file):
    """
    Copy file with destination is local folder
    :param src_file: source file, can be local or HDFS
    :param dst_file: destination file on local filesystem
    """
    if src_file.startswith(hdfs_protocol):
        hdfs_copy_file_to_local(src_file, dst_file)
    elif src_file.startswith(s3_protocol):
        print '[ERRR] fs_copy_file_to_local: not supported file system'
    else:
        local_copy_file(src_file, dst_file)


def fs_make_folder(full_path):
    """
    Make folder. Work with local and HDFS
    :param full_path: folder path to create
    """
    if full_path.startswith(hdfs_protocol):
        hdfs_make_folder(full_path)
    elif full_path.startswith(s3_protocol):
        print '[ERRR] fs_make_folder: not supported file system'
    else:
        local_make_folder(full_path)


def fs_remove_file_pattern(full_path, name_pattern):
    """
    Remove files in folder not recursively
    :param full_path: folder path
    :param name_pattern: name pattern of files to be removed
    """
    if full_path.startswith(hdfs_protocol):
        hdfs_remove_folder(os.path.join(full_path, name_pattern))
    elif full_path.startswith(s3_protocol):
        print '[ERRR] fs_remove_file_pattern: not supported file system'
    else:
        local_remove_file_pattern(full_path, name_pattern)


def fs_path_climb_up(full_path, level_up):
    """
    Calculate ancestor folder
    :param full_path: input path
    :param level_up: number of level to go up from input path
    :return: path to ancestor folder
    """
    lu = level_up
    fp = full_path
    while lu > 0:
        fp_parent = os.path.split(fp)[0]
        if fp_parent == '':
            break
        else:
            fp = fp_parent
            lu -= 1
    return fp


def fs_remove_folder(full_path):
    """
    Remove folder. Work with local or HDFS
    :param full_path: path to folder to remove
    """
    if full_path.startswith(hdfs_protocol):
        hdfs_remove_folder(full_path)
    elif full_path.startswith(s3_protocol):
        print '[ERRR] fs_remove_folder: not supported file system'
    else:
        local_remove_folder(full_path)


def fs_convert_path_to_posix(full_path):
    """
    Convert input path to using `/` as delimiter
    :param full_path: input path
    :return: Posix path
    """
    path_elements = local_split_path(full_path)
    if len(path_elements) < 1:
        return full_path
    path_folders = [e[1] for e in path_elements]
    path_folders.insert(0, path_elements[0][0])
    path_posix = '/'.join(path_folders)
    return path_posix


def fs_normalize_s3_path(s3_path):
    """
    Normalize S3 path
    :param s3_path: S3 path
    :return: Normalized S3 path with `S3://` added and path is in Posix format
    """
    if s3_path.startswith(s3_protocol):
        s3_path_posix = s3_protocol + fs_convert_path_to_posix(s3_path.replace(s3_protocol, ''))
        return s3_path_posix
    else:
        return s3_path


def add_uuid_to_file_name(file_name):
    """
    Adding uuid to end of file name to make file unique
    :param file_name: input filename
    :return: new filename
    """
    rt, ext = os.path.splitext(file_name)
    rt = rt + '_' + uuid.uuid4().urn[9:]
    if ext != '':
        rt = rt + ext
    return rt
