def dictionary_merger(dictionary_1: dict, dictionary_2: dict):
    if dictionary_1 is None:
        dictionary_1 = {}
        for key in dictionary_2.keys():
            dictionary_1[key] = [dictionary_2[key]]
        return dictionary_1

    else:
        for key in dictionary_1.keys():
            dictionary_1[key] += [dictionary_2[key]]
        return dictionary_1
