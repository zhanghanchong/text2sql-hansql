#coding=utf8
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField
from asdl.sql.unparser.unparser_base import UnParser


class UnParserV3(UnParser):
    def unparse_sql_unit(self, sql_field: RealizedField, db: dict, *args, **kargs):
        sql_ast = sql_field.value
        from_field, select_field, where_field, groupby_field, orderby_field = sql_ast.fields
        from_str = 'FROM ' + self.unparse_from(from_field, db, *args, **kargs)
        select_str = 'SELECT ' + self.unparse_select(select_field, db, *args, **kargs)
        where_str, groupby_str, orderby_str = '', '', ''
        if where_field.value.production.constructor.name != 'CondReduce':
            where_str = 'WHERE ' + self.unparse_where(where_field, db, *args, **kargs)
        if groupby_field.value.production.constructor.name != 'GroupByReduce':
            groupby_str = 'GROUP BY ' + self.unparse_groupby(groupby_field, db, *args, **kargs)
        if orderby_field.value.production.constructor.name != 'OrderByReduce':
            orderby_str = 'ORDER BY ' + self.unparse_orderby(orderby_field, db, *args, **kargs)
        return ' '.join([select_str, from_str, where_str, groupby_str, orderby_str])

    def unparse_select(self, select_field: RealizedField, db: dict, *args, **kargs):
        select_ast = select_field.value
        select_list = select_ast.fields
        select_items = []
        for val_unit_field in select_list:
            val_unit_str = self.unparse_val_unit(val_unit_field.value, db, *args, **kargs)
            select_items.append(val_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_field: RealizedField, db: dict, *args, **kargs):
        from_ast = from_field.value
        ctr_name = from_ast.production.constructor.name
        if 'Table' in ctr_name:
            tab_names = []
            for tab_field in from_ast.fields:
                tab_name = db['table_names_original'][int(tab_field.value)]
                tab_names.append(tab_name)
            return ' JOIN '.join(tab_names)
        else:
            return '( ' + self.unparse_sql(from_ast.fields[0].value, db, *args, **kargs) + ' )'

    def unparse_groupby(self, groupby_field: RealizedField, db: dict, *args, **kargs):
        groupby_ast = groupby_field.value
        groupby_str = []
        num = len(groupby_ast.fields) - 1
        for col_id_field in groupby_ast.fields[:num]:
            col_name = self.unparse_col_unit(col_id_field.value, db, *args, **kargs)
            groupby_str.append(col_name)
        groupby_str = ' , '.join(groupby_str)
        if groupby_ast.fields[-1].value.production.constructor.name != 'CondReduce':
            having = groupby_ast.fields[-1].value
            having_str = self.unparse_conds(having, db, *args, **kargs)
            return groupby_str + ' HAVING ' + having_str
        else:
            return groupby_str

    def unparse_orderby(self, orderby_field: RealizedField, db: dict, *args, **kargs):
        orderby_ast = orderby_field.value
        ctr_name = orderby_ast.production.constructor.name.lower()
        val_unit_str = []
        for val_unit_field in orderby_ast.fields:
            val_unit_ast = val_unit_field.value
            val_unit_str.append(self.unparse_col_unit(val_unit_ast, db, *args, **kargs))
        val_unit_str = ' , '.join(val_unit_str)
        if 'asc' in ctr_name and 'limit' in ctr_name:
            return '%s ASC LIMIT 1' % (val_unit_str)
        elif 'asc' in ctr_name:
            return '%s ASC' % (val_unit_str)
        elif 'desc' in ctr_name and 'limit' in ctr_name:
            return '%s DESC LIMIT 1' % (val_unit_str)
        else:
            return '%s DESC' % (val_unit_str)

    def unparse_conds(self, conds_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = conds_ast.production.constructor.name
        if ctr_name == 'CondReduce':
            return ''
        if ctr_name in ['And', 'Or']:
            left_cond, right_cond = conds_ast.fields
            left_str = self.unparse_conds(left_cond.value, db, *args, **kargs)
            right_str = self.unparse_conds(right_cond.value, db, *args, **kargs)
            if left_str == '':
                return right_str
            if right_str == '':
                return left_str
            return left_str + ' ' + ctr_name.upper() + ' ' + right_str
        return self.unparse_cond(conds_ast, db, *args, **kargs)
