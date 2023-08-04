def example_objective_function(genome):
    
    genome_sum = 0
    
    for dim in genome:
        genome_sum += dim

    return genome_sum

def example_behavior_function(genome):

    bhv_vector = [None,None]
    
    # nothing meaningful being done, just calculates "random" behavioral values within value range from given genome input
    bhv_vector[0] = 0
    for dim in genome:
        bhv_vector[0] += dim*3.1415 % 100 # generates value between 0 and 100

    bhv_vector[1] = 0
    for dim in genome:
        bhv_vector[1] += dim*2.7182 % 20 # generates value between 0 and 20

    return bhv_vector

def example_variation_function(genomes):
    
    for genome in genomes:
        i=0
        for dim in genome:
            if i%2 == 0:
                dim += -1
            else:
                dim -=  1
            i += 1

    return genomes