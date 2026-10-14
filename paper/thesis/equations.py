from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def sequence(parent, values):
    for value in values:
        if isinstance(value, str):
            run = OxmlElement('m:r')
            text = OxmlElement('m:t')
            text.set(qn('xml:space'), 'preserve')
            text.text = value
            run.append(text)
            parent.append(run)
        elif isinstance(value, list):
            sequence(parent, value)
        else:
            parent.append(value)
    return parent


def group(tag, *values):
    return sequence(OxmlElement('m:' + tag), values)


def sub(base, index):
    return group('sSub', group('e', base), group('sub', index))


def sup(base, power):
    return group('sSup', group('e', base), group('sup', power))


def frac(numerator, denominator):
    return group('f', group('num', numerator), group('den', denominator))


def total(index, expression):
    node = OxmlElement('m:nary')
    props = OxmlElement('m:naryPr')
    for tag, value in [('chr', '∑'), ('limLoc', 'subSup'), ('supHide', '1')]:
        prop = OxmlElement('m:' + tag)
        prop.set(qn('m:val'), value)
        props.append(prop)
    return sequence(node, [props, group('sub', index), group('sup'), group('e', expression)])


def equation(number):
    if number == '3-1':
        parts = ['T = round(', sub('f', 's'), ' × Δ) = 100']
    elif number == '3-2':
        parts = [sub('L', 'read'), ' = ', sub('n', 'char'), ' × T,   ', sub('L', 'inner'), ' = (', sub('n', 'char'), ' + 1) × T']
    elif number == '4-1':
        f = lambda: [sub('f', 'θ'), '(', sub('X', 'i'), ', ', sub('M', 'i'), ')']
        parts = [sub('e', 'i'), ' = ', frac(f(), sub(['‖', f(), '‖'], '2'))]
    elif number == '4-2':
        g = lambda: ['g(', sub('s', 'i'), ')']
        parts = [sub('t', 'i'), ' = ', frac(g(), sub(['‖', g(), '‖'], '2')), ',   ', sub('a', 'ij'), ' = ', frac([sup(sub('e', 'i'), 'T'), sub('t', 'j')], 'τ')]
    elif number == '4-3':
        parts = [sup(sub('H', 'i'), '(0)'), ' = [', sub('h', 'i1'), ', …, ', sub('h', 'in'), '] + ', sub('P', '1:n')]
    elif number == '4-4':
        parts = [sub('H', 'i'), ' = Transformer(', sup(sub('H', 'i'), '(0)'), ', ', sub('M', 'i'), ')']
    elif number == '4-5':
        parts = [sub('u', 'i'), ' = MultiHeadAttention(q, ', sub('H', 'i'), ', ', sub('H', 'i'), '; ', sub('M', 'i'), ')']
    elif number == '4-6':
        parts = [sub('e', 'i'), ' = Normalize(', sub('W', 'out'), sub('u', 'i'), ' + ', sub('b', 'out'), ')']
    elif number == '4-7':
        parts = [sub('u', 'i'), ' = ', frac(total('j', [sub('M', 'ij'), sub('H', 'ij')]), total('j', sub('M', 'ij')))]
    elif number == '4-8':
        parts = [sub('P', 'ij'), ' = 1[', sub('G', 'i'), ' = ', sub('G', 'j'), ']']
    elif number == '4-9':
        ratio = frac(['exp(', sub('a', 'ij'), ')'], total('k', ['exp(', sub('a', 'ik'), ')']))
        parts = [sub('ℓ', 'i'), ' = −', frac('1', ['|', sub('P', 'i'), '|']), total(['j∈', sub('P', 'i')], ['log[', ratio, ']'])]
    elif number == '4-10':
        parts = ['L = ', frac('1', '2'), '[', sub('Mean', 'i'), sub('ℓ', 'i'), '(A) + ', sub('Mean', 'i'), sub('ℓ', 'i'), '(', sup('A', 'T'), ')]']
    elif number == '4-11':
        parts = ['P(T→E@k) = 1 − ', frac('C(N−m, k)', 'C(N, k)')]
    elif number == '4-12':
        parts = ['w = ', sup(['(', sup(sub('Z', 'train'), 'T'), sub('Z', 'train'), ' + αI)'], '−1'), sup(sub('Z', 'train'), 'T'), sub('y', 'centered')]
    elif number == '4-13':
        numerator = total('i', sup(['(', sub('y', 'i'), '−', sub('ŷ', 'i'), ')'], '2'))
        denominator = total('i', sup(['(', sub('y', 'i'), '−', sub('ȳ', 'val'), ')'], '2'))
        parts = [sup('R', '2'), ' = 1 − ', frac(numerator, denominator)]
    else:
        raise ValueError(number)
    return group('oMath', *parts)
