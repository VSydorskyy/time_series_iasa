def read_data_from_file(file_path):
    with open(file_path) as file:
        content = file.read() 
        
    content = content.split('\n')
    return list(map(float, content))