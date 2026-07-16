package agents

func normalizeBareNoneJSONLiterals(raw []byte) ([]byte, []string) {
	if len(raw) == 0 {
		return raw, nil
	}
	out := make([]byte, 0, len(raw))
	inString := false
	escaped := false
	changed := false
	for i := 0; i < len(raw); {
		b := raw[i]
		if inString {
			out = append(out, b)
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				inString = false
			}
			i++
			continue
		}
		if b == '"' {
			inString = true
			out = append(out, b)
			i++
			continue
		}
		if isBareNoneJSONValue(raw, i) {
			out = append(out, "null"...)
			i += len("none")
			changed = true
			continue
		}
		out = append(out, b)
		i++
	}
	if !changed {
		return raw, nil
	}
	return out, []string{"bare none literal converted to JSON null"}
}

func isBareNoneJSONValue(raw []byte, index int) bool {
	if index+len("none") > len(raw) {
		return false
	}
	if lowerASCII(raw[index]) != 'n' || lowerASCII(raw[index+1]) != 'o' || lowerASCII(raw[index+2]) != 'n' || lowerASCII(raw[index+3]) != 'e' {
		return false
	}
	prev := previousNonJSONWhitespace(raw, index-1)
	next := nextNonJSONWhitespace(raw, index+len("none"))
	return canPrecedeJSONValue(prev) && canFollowJSONValue(next)
}

func previousNonJSONWhitespace(raw []byte, index int) byte {
	for i := index; i >= 0; i-- {
		if !isJSONWhitespace(raw[i]) {
			return raw[i]
		}
	}
	return 0
}

func nextNonJSONWhitespace(raw []byte, index int) byte {
	for i := index; i < len(raw); i++ {
		if !isJSONWhitespace(raw[i]) {
			return raw[i]
		}
	}
	return 0
}

func isJSONWhitespace(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}

func lowerASCII(b byte) byte {
	if b >= 'A' && b <= 'Z' {
		return b + ('a' - 'A')
	}
	return b
}

func canPrecedeJSONValue(b byte) bool {
	return b == 0 || b == ':' || b == ',' || b == '['
}

func canFollowJSONValue(b byte) bool {
	return b == 0 || b == ',' || b == ']' || b == '}'
}
