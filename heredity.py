import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    # have_trait is a list
    for have_trait in powerset(names):
        '''
        # Check if current set of people violates known information
        # fails_evidence is True if:
        #   There is a person in people who has a trait value of True and ISN'T in have_trait
        #   OR there is a person in people who has a trait value of Fale and IS in have_trait
        '''
        # BASICALLY this section ensures that have_trait is a list containing
        #   All the people who have been identified as having the trait
        #   AND all the people about whom we are not sure if they have the trait
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        # If fails evidence, skip directly to to the next possible set of have_trait
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probability = 1
    
    # Calculates the probability of the genetics occuring as described by the one_gene and two_gene lists
    for person in people:
        if person in one_gene:
            if people[person]["mother"] == None:
                probability *= PROBS["gene"][1]
            else:
                probability *= probability_of_one_gene(person, people, one_gene, two_genes)
        elif person in two_genes:
            if people[person]["mother"] == None:
                probability *= PROBS["gene"][2]
            else:
                probability *= probability_of_two_genes(person, people, one_gene, two_genes)
        else:
            if people[person]["mother"] == None:
                probability *= PROBS["gene"][0]
            else:
                probability *= probability_of_no_genes(person, people, one_gene, two_genes)

    for person in have_trait:
        if person in one_gene:
            probability *= PROBS["trait"][1][True]
        elif person in two_genes:
            probability *= PROBS["trait"][2][True]
        else:
            probability *= PROBS["trait"][0][True]

    for person in people:
        if person not in have_trait:
            if person in one_gene:
                probability *= PROBS["trait"][1][False]
            elif person in two_genes:
                probability *= PROBS["trait"][2][False]
            else:
                probability *= PROBS["trait"][0][False]


    return probability

def probability_of_no_genes(person, people, one_gene, two_genes):
    # The only way a person can have no mutated genes is if both of the parents pass on an unmutated gene
    #   (having already taken into account random mutations)
    return (1-probability_child_inherit_gene(people[person]["mother"], one_gene, two_genes)) * (1-probability_child_inherit_gene(people[person]["father"], one_gene, two_genes))

def probability_of_one_gene(person, people, one_gene, two_genes):
    # There are two ways a person can have only one gene:
    #   Their mother passes on the gene and their father doesn't
    #   Their father passes on the gene and their mother doesn't
    scenario1 = probability_child_inherit_gene(people[person]["mother"], one_gene, two_genes) * (1-probability_child_inherit_gene(people[person]["father"], one_gene, two_genes))
    scenario2 = probability_child_inherit_gene(people[person]["father"], one_gene, two_genes) * (1-probability_child_inherit_gene(people[person]["mother"], one_gene, two_genes))
    return scenario1 + scenario2

def probability_of_two_genes(person, people, one_gene, two_genes):
    # There is only one way a person can have two genes, both of their parents passed it down
    return probability_child_inherit_gene(people[person]["mother"], one_gene, two_genes) * probability_child_inherit_gene(people[person]["father"], one_gene, two_genes)

def probability_child_inherit_gene(person, one_gene, two_genes):
    '''
    # If we don't know who the parent is (i.e. we don't know if they have 0, 1, or 2 mutated genes)...
    if person == None:
        # Three options: parent has 0, 1, or 2 genes
        # If they have 0 genes, the only way they pass on a mutated gene is if the one passed on actually mutates
        inheritance_probability = PROBS["gene"][0] * PROBS["mutation"]
        # If they have 1 gene, they can pass on a mutated gene by:
            # Initialy passing on a mutated gene, and that gene not mutating
            # OR initially passing on a normal gene and that one mutating
        inheritance_probability += PROBS["gene"][1] * 0.5 * (1-PROBS["mutation"])
        inheritance_probability += PROBS["gene"][1] * 0.5 * PROBS["mutation"]
        # If they have 2 genes, they can pass on a mutated gene by passing either of their genes and neither of them mutating
        inheritance_probability += PROBS["gene"][2] * (1-PROBS["mutation"])
    '''
    if person in one_gene:
        # If the parent is assumed to have one mutated gene then there are two ways a child can inherit a mutated gene:
            # They inherit the normal gene and it mutates
            # They inherit the mutated gene and it does not mutate
        inheritance_probability = (0.5 * PROBS["mutation"]) + (0.5 * (1 - PROBS["mutation"]))
    
    elif person in two_genes:
        # If the parent is assumed to have two mutates genes then there is really one way a child can inherit a mutated gene:
            # They inherit either of the mutated genes and it does not mutate
        inheritance_probability = 1-PROBS["mutation"]
    
    else:
        # If the parent is assumed to have no mutated genes then there is really one way a child can inherit a mutated gene:
            # They inherit either normal gene and it mutates
        inheritance_probability = PROBS["mutation"]
    
    return inheritance_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Updating the gene distribution
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:            
            probabilities[person]["gene"][0] += p

        #Updating the trait distribution
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        geneSum = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        probabilities[person]["gene"][0] /= geneSum
        probabilities[person]["gene"][1] /= geneSum
        probabilities[person]["gene"][2] /= geneSum

        traitSum = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]

        probabilities[person]["trait"][True] /= traitSum
        probabilities[person]["trait"][False] /= traitSum


if __name__ == "__main__":
    main()
