def read_data_from_file(file_path):
    with open(file_path) as file:
        content = file.read() 
        
    content = content.split('\n')
    content = list(filter(lambda x: x != '', content))
    return list(map(lambda x: float(x.strip()), content))