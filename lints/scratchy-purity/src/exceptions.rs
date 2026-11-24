//! The purity exceptions, `lints/scratchy-purity/exceptions.toml`: the exceptions the purity
//! skill itself names, one `[[exception]]` table each.

/// The exceptions file, relative to the workspace root.
pub const FILE: &str = "lints/scratchy-purity/exceptions.toml";

/// One exception: the item it exempts, as a crate-qualified path (`krate::module::Type::method`),
/// the rule it is an exception to, and who approved it.
#[derive(Debug, PartialEq, Eq)]
pub struct Exception {
    pub rule: String,
    pub item: String,
    pub approved: String,
}

/// The exceptions in `text`, or what is wrong with it. Every entry names its rule, its item and
/// its approval; an unknown key is an error, so a misspelt field never passes as an exception.
pub fn parse(text: &str) -> Result<Vec<Exception>, String> {
    let table: toml::Table = text
        .parse()
        .map_err(|e: toml::de::Error| e.message().to_owned())?;
    if let Some(key) = table.keys().find(|k| *k != "exception") {
        return Err(format!("unknown top-level key `{key}`"));
    }
    let Some(entries) = table.get("exception") else {
        return Ok(Vec::new());
    };
    let entries = entries
        .as_array()
        .ok_or("`exception` is not an array of tables")?;
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let entry = entry
                .as_table()
                .ok_or(format!("exception {} is not a table", i + 1))?;
            if let Some(key) = entry
                .keys()
                .find(|k| !["rule", "item", "approved"].contains(&k.as_str()))
            {
                return Err(format!("exception {} has an unknown key `{key}`", i + 1));
            }
            let field = |name: &str| {
                entry
                    .get(name)
                    .and_then(toml::Value::as_str)
                    .filter(|v| !v.trim().is_empty())
                    .map(str::to_owned)
                    .ok_or(format!("exception {} has no `{name}`", i + 1))
            };
            Ok(Exception {
                rule: field("rule")?,
                item: field("item")?,
                approved: field("approved")?,
            })
        })
        .collect()
}

/// Whether `path` (crate-qualified) is the item `exempt` names, or inside it.
pub fn covers(exempt: &str, path: &str) -> bool {
    path.strip_prefix(exempt)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with("::"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_file_has_no_exceptions() {
        assert_eq!(parse("# only comments\n"), Ok(Vec::new()));
    }

    #[test]
    fn each_entry_names_its_rule_item_and_approval() {
        let text = r#"
[[exception]]
rule = "P7"
item = "a::registry::Row::claims"
approved = "the purity skill, P4 exception 1"

[[exception]]
rule = "P2"
item = "b::lut"
approved = "the user"
"#;
        assert_eq!(
            parse(text),
            Ok(vec![
                Exception {
                    rule: "P7".into(),
                    item: "a::registry::Row::claims".into(),
                    approved: "the purity skill, P4 exception 1".into(),
                },
                Exception {
                    rule: "P2".into(),
                    item: "b::lut".into(),
                    approved: "the user".into(),
                },
            ])
        );
    }

    #[test]
    fn a_malformed_entry_is_an_error() {
        let missing = "[[exception]]\nrule = \"P7\"\nitem = \"a::b\"\n";
        assert_eq!(parse(missing), Err("exception 1 has no `approved`".into()));
        let blank = "[[exception]]\nrule = \"P7\"\nitem = \" \"\napproved = \"x\"\n";
        assert_eq!(parse(blank), Err("exception 1 has no `item`".into()));
        let typo = "[[exception]]\nrule = \"P7\"\nitme = \"a\"\nitem = \"a\"\napproved = \"x\"\n";
        assert_eq!(
            parse(typo),
            Err("exception 1 has an unknown key `itme`".into())
        );
        assert_eq!(
            parse("[exceptions]\n"),
            Err("unknown top-level key `exceptions`".into())
        );
        assert_eq!(
            parse("exception = 1\n"),
            Err("`exception` is not an array of tables".into())
        );
        assert!(parse("[[exception]\n").is_err());
    }

    #[test]
    fn an_exception_covers_its_item_and_what_is_inside_it() {
        assert!(covers("a::Row::claims", "a::Row::claims"));
        assert!(covers("a::Row", "a::Row::claims"));
        assert!(!covers("a::Row::claims", "a::Row::claims_all"));
        assert!(!covers("a::Row::claims", "a::Row"));
    }
}
