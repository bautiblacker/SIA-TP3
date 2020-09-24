class FileParser:

        def __entries_normalizer(entries, max_values, min_values):
            for row in  entries:
                for idx in range(len(row)):
                    row[idx] = (row[idx] - min_values[idx]) / (max_values[idx] - min_values[idx])

            return entries

        
        def _outputs_normalizer(outputs, max_value, min_value):
            for idx in  range(len(outputs)):
                outputs[idx] = (outputs[idx] - min_value) / (max_value - min_value)

            return outputs

        @staticmethod
        def entries_parser():
            entries_file = open("/Users/nachograsso/Desktop/ITBA/SIA/SIA-TP3/SIA-TP3/conjunto-entrenamiento-ej2.txt", "r")
            entries = []
            max_values = [0,0,0]
            min_values = [1000,1000,1000]
            for l_i in entries_file:
                entry = list(map(float, l_i.split()))
                entries.append(entry)
                for idx in range(len(entry)):
                    max_values[idx] = max(max_values[idx], entry[idx])
                    min_values[idx] = min(min_values[idx], entry[idx]) 

            return FileParser.__entries_normalizer(entries, max_values, min_values)


        @staticmethod
        def outputs_parser():
            output_file = open("/Users/nachograsso/Desktop/ITBA/SIA/SIA-TP3/SIA-TP3/salida-esperada-ej2.txt", "r")
            outputs = []
            max_value = 0
            min_value = 10000
            for l_i in output_file:
                output = float(l_i.split()[0])
                outputs.append(output)
                max_value = max(max_value, output)
                min_value = min(min_value, output) 

            return FileParser._outputs_normalizer(outputs, max_value, min_value)


        @staticmethod
        def mlp_entries_parser(rows, cols):
            entries_file = open('SIA-TP3/TP3-ej3-mapa-de-pixeles-digitos-decimales.txt')
            entries = []
            count = 0
            e = ''
            for l_i in entries_file:
                entry = l_i.split()
                if (count + len(entry)) <= rows * cols:
                    for i in range(len(entry)):
                        e += (entry[i] + ' ')
                        count +=1
                if count >= 35:
                    entries.append(e)
                    count = 0
                    e = ''
            return entries
